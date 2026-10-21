"""RaPD mainline text-to-image training and validation."""

import os
import copy
import time
import random
import argparse
import importlib
from contextlib import nullcontext

import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from src.utils.patch_bugs import *  # noqa: F403 — NCCL env vars, warning suppression
from src.models.autoencoder.base import fp2uint8
from src.utils.no_grad import no_grad, filter_nograd_tensors

from src.callbacks.simple_ema import SimpleEMA
from src.callbacks.save_images import SaveImagesHook
from src.models.autoencoder.coordinates import make_coord_cell_grid
from src.models.autoencoder.cqar_ae import (
    CQARAutoencoder,
)
from src.models.transformer.JiT_T2I import JiT_T2I
from src.models.conditioner.qwen3_text_encoder import Qwen3TextEncoder
from src.diffusion.flow_matching.sampling import EulerSamplerJiT, ode_step_fn
from src.diffusion.flow_matching.scheduling import LinearScheduler
from src.diffusion.base.guidance import simple_guidance_fn
from torch.optim import AdamW
from src.utils.lr_scheduler import ConstantWithWarmup


torch._functorch.config.donated_buffer = False


# ── distributed helpers ──────────────────────────────────────────────────────


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_distributed():
    def _set_cuda_device(local_rank):
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        device_count = torch.cuda.device_count()
        if local_rank >= device_count:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but torch only sees {device_count} CUDA device(s). "
                f"CUDA_VISIBLE_DEVICES={visible_devices!r}. "
                "Make sure --nproc_per_node matches the number of visible GPUs."
            )
        try:
            torch.cuda.set_device(local_rank)
        except RuntimeError as exc:
            mapped_device = None
            if visible_devices:
                devices = [d.strip() for d in visible_devices.split(",")]
                if local_rank < len(devices):
                    mapped_device = devices[local_rank]
            mapped = f" mapped CUDA_VISIBLE_DEVICES entry={mapped_device!r}" if mapped_device else ""
            raise RuntimeError(
                f"Failed to set CUDA device for LOCAL_RANK={local_rank}{mapped}. "
                f"CUDA_VISIBLE_DEVICES={visible_devices!r}. "
                "The mapped GPU may be busy, unavailable, hidden by the scheduler, or in exclusive mode."
            ) from exc

    if "LOCAL_RANK" not in os.environ:
        if torch.cuda.is_available():
            _set_cuda_device(0)
        return 0, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    _set_cuda_device(local_rank)
    dist.init_process_group("nccl")
    return local_rank, rank, world_size


def cleanup_distributed(world_size):
    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def all_gather(tensor, world_size):
    if world_size == 1:
        return tensor
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)


# ── data helpers ─────────────────────────────────────────────────────────────


def collate_fn(batch):
    x, y, metadata = list(zip(*batch))
    stacked = {}
    for key in metadata[0].keys():
        try:
            if isinstance(metadata[0][key], torch.Tensor):
                stacked[key] = torch.stack([m[key] for m in metadata])
            else:
                stacked[key] = [m[key] for m in metadata]
        except Exception:
            pass
    return torch.stack(x), y, stacked


def micro_batch_collate_fn(batch):
    flat = []
    for micro in batch:
        flat.extend(micro)
    return collate_fn(flat)


def eval_collate_fn(batch):
    x, y, metadata = list(zip(*batch))
    return torch.stack(x), y, metadata


def make_train_loader(dataset, batch_size, num_workers, rank, world_size, prefetch=8):
    micro = getattr(dataset, "micro_batch_size", None)
    if micro:
        assert batch_size % micro == 0
        batch_size = batch_size // micro
        cfn = micro_batch_collate_fn
    else:
        cfn = collate_fn
    sampler = (
        None
        if isinstance(dataset, IterableDataset)
        else DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    )
    extra = {"prefetch_factor": prefetch} if num_workers > 0 else {}
    return DataLoader(
        dataset, batch_size, timeout=6000,
        num_workers=num_workers,
        collate_fn=cfn, sampler=sampler, **extra,
    )


def make_eval_loader(dataset, batch_size, num_workers, rank, world_size):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    extra = {"prefetch_factor": 2} if num_workers > 0 else {}
    return DataLoader(
        dataset, batch_size,
        num_workers=num_workers,
        collate_fn=eval_collate_fn, sampler=sampler, **extra,
    )


def move_to_device(batch, device):
    x, y, metadata = batch
    x = x.to(device, non_blocking=True)
    if isinstance(y, torch.Tensor):
        y = y.to(device, non_blocking=True)

    def _move_dict(d):
        return {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in d.items()
        }

    if isinstance(metadata, dict):
        metadata = _move_dict(metadata)
    elif isinstance(metadata, (list, tuple)):
        metadata = type(metadata)(_move_dict(m) if isinstance(m, dict) else m for m in metadata)
    return x, y, metadata


def metadata_scalar(metadata, key, index=0):
    if isinstance(metadata, dict):
        value = metadata.get(key)
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.dim() == 0:
                return value.item()
            return value[index].item()
        if isinstance(value, (list, tuple)):
            return value[index]
        return value
    if isinstance(metadata, (list, tuple)):
        if index >= len(metadata) or not isinstance(metadata[index], dict):
            return None
        value = metadata[index].get(key)
        if isinstance(value, torch.Tensor):
            return value.item()
        return value
    return None


def metadata_render_size(metadata, index=0):
    height = metadata_scalar(metadata, "render_height", index)
    width = metadata_scalar(metadata, "render_width", index)
    if height is None or width is None:
        height = metadata_scalar(metadata, "source_original_height", index)
        width = metadata_scalar(metadata, "source_original_width", index)
    if height is None or width is None:
        return None
    return (int(height), int(width))


def metadata_render_sizes(metadata, batch_size):
    sizes = [metadata_render_size(metadata, idx) for idx in range(batch_size)]
    if any(size is None for size in sizes):
        return None
    return sizes


def sampler_forward(sampler, net, noise, condition, uncondition):
    return sampler(net, noise, condition, uncondition)


def sampler_forward_with_trajectories(sampler, net, noise, condition, uncondition):
    return sampler(net, noise, condition, uncondition, return_x_trajs=True)


def infer_default_decode_size(data_cfg):
    for ds_key in ("train_dataset", "eval_dataset", "pred_dataset"):
        init_args = data_cfg.get(ds_key, {}).get("init_args", {})
        resolution = init_args.get("resolution")
        if isinstance(resolution, int) and resolution > 0:
            return (resolution, resolution)
    return None


def instantiate_from_config(cfg, default_class_path=None):
    cfg = cfg or {}
    class_path = cfg.get("class_path", default_class_path)
    if not class_path:
        raise ValueError("Missing class_path in config")
    module_name, class_name = class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(**cfg.get("init_args", {}))


def build_denoisers(config, device, is_training, eval_original=False):
    """Build both networks for training, or only the selected inference network."""
    net = instantiate_from_config(config, "src.models.transformer.JiT_T2I.JiT_T2I").to(device)
    if is_training:
        return net, copy.deepcopy(net)
    return (net, None) if eval_original else (None, net)


def parse_decode_size(value):
    if value is None:
        return None
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("decode_size must be positive")
        return (value, value)
    if isinstance(value, str):
        value = value.strip().lower()
        if not value:
            return None
        if value.isdigit():
            size = int(value)
            if size <= 0:
                raise ValueError("decode_size must be positive")
            return (size, size)
        if "x" in value:
            parts = value.split("x")
            if len(parts) == 2 and all(part.strip().isdigit() for part in parts):
                height, width = (int(part.strip()) for part in parts)
                if height <= 0 or width <= 0:
                    raise ValueError("decode_size dimensions must be positive")
                return (height, width)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        height, width = int(value[0]), int(value[1])
        if height <= 0 or width <= 0:
            raise ValueError("decode_size dimensions must be positive")
        return (height, width)
    raise ValueError("decode_size must be an int, a [height, width] pair, or a 'HxW' string")


def parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Invalid boolean value: {value}")
    return bool(value)


def parse_inference_dtype(value):
    if value is None:
        value = "bf16"
    if isinstance(value, torch.dtype):
        if value in (torch.bfloat16, torch.float16):
            return value
        if value == torch.float32:
            return None
        raise ValueError(f"Unsupported inference dtype: {value}")
    if not isinstance(value, str):
        raise ValueError(f"Unsupported inference dtype: {value}")

    lowered = value.strip().lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16", "half"}:
        return torch.float16
    if lowered in {"fp32", "float32", "full", "none", "off"}:
        return None
    raise ValueError(
        "model.inference_dtype must be one of: bf16, bfloat16, fp16, "
        "float16, fp32, float32"
    )


def parse_intermediate_eval_cfg(value):
    cfg = value or {}
    if isinstance(cfg, bool):
        cfg = {"enabled": cfg}
    return {
        "enabled": parse_bool(cfg.get("enabled"), default=False),
        "save_latent_gray": parse_bool(cfg.get("save_latent_gray"), default=True),
        "save_rendered": parse_bool(cfg.get("save_rendered"), default=True),
        "subdir": str(cfg.get("subdir", "intermediates")),
    }


def inference_autocast(device, dtype):
    if dtype is None or torch.device(device).type != "cuda":
        return nullcontext()
    return torch.autocast("cuda", dtype=dtype)


def inference_dtype_label(dtype):
    if dtype is None:
        return "fp32"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    return str(dtype)


def renderer_active_blocks(ae):
    renderer = getattr(ae, "renderer", None)
    if renderer is None or not hasattr(renderer, "get_active_num_blocks"):
        return None, None
    active, total = renderer.get_active_num_blocks()
    if active is None or total is None:
        return None, None
    return int(active), int(total)


def resolve_render_fast_blocks(value, total_blocks, command, decode_size):
    if total_blocks is None or total_blocks <= 0:
        return None

    if value is None:
        return None

    if isinstance(value, bool):
        return max(1, (total_blocks + 1) // 2) if value else None
    if isinstance(value, int):
        if value <= 0:
            return None
        return min(value, total_blocks)
    if not isinstance(value, str):
        raise ValueError(f"Unsupported model.render_fast_blocks value: {value}")

    lowered = value.strip().lower()
    if lowered in {"", "0", "false", "off", "none", "full"}:
        return None
    if lowered in {"1", "true", "on", "min", "minimum"}:
        return 1
    if lowered == "auto":
        if command != "predict" or decode_size is None or max(decode_size) < 1024:
            return None
        return (
            max(1, (total_blocks + 3) // 4)
            if max(decode_size) >= 2048
            else max(1, (total_blocks + 1) // 2)
        )
    if lowered == "half":
        return max(1, (total_blocks + 1) // 2)
    if lowered in {"quarter"}:
        return max(1, (total_blocks + 3) // 4)
    if lowered.isdigit():
        return max(0, min(int(lowered), total_blocks))
    raise ValueError(
        "model.render_fast_blocks must be an int, true/false, "
        "or one of: auto, half, quarter, min, full/off"
    )


def configure_renderer_fast_blocks(ae, value, command, decode_size, is_zero):
    renderer = getattr(ae, "renderer", None)
    active, total = renderer_active_blocks(ae)
    if renderer is None or total is None:
        if is_zero and value not in (None, False, 0, "0", "false", "off", "none", "full"):
            print("[Warning] model.render_fast_blocks was set but this AE renderer does not support it")
        return None

    fast_blocks = resolve_render_fast_blocks(value, total, command, decode_size)
    if fast_blocks is None or fast_blocks >= total:
        if is_zero and value not in (None, False, 0, "0", "false", "off", "none", "full"):
            print(f"[Info] Renderer fast blocks disabled ({active}/{total} blocks active)")
        return None

    renderer.set_active_num_blocks(fast_blocks)
    if is_zero:
        source = "auto" if value is None else value
        print(
            "[Info] Renderer fast blocks enabled: "
            f"{fast_blocks}/{total} active (model.render_fast_blocks={source})"
        )
    return fast_blocks


# ── checkpoint helpers ───────────────────────────────────────────────────────


def save_ckpt(path, denoiser, ema_denoiser, diff_trainer, optimizer, lr_sched, step):
    sd = {}
    denoiser.state_dict(destination=sd, prefix="denoiser.")
    ema_denoiser.state_dict(destination=sd, prefix="ema_denoiser.")
    diff_trainer.state_dict(destination=sd, prefix="diffusion_trainer.")
    ckpt = {"state_dict": sd, "optimizer": optimizer.state_dict(), "global_step": step}
    if lr_sched is not None:
        ckpt["lr_scheduler"] = lr_sched.state_dict()
    torch.save(ckpt, path)


def load_ckpt(path, denoiser, ema_denoiser, diff_trainer, device, pos_embed_load_policy="drop"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "state_dict" not in ckpt and "model" in ckpt:
        ckpt["state_dict"] = ckpt["model"]
    if "state_dict" not in ckpt:
        return ckpt

    sd = ckpt["state_dict"]
    current_sd = {}
    if denoiser is not None:
        denoiser.state_dict(destination=current_sd, prefix="denoiser.")
    if ema_denoiser is not None:
        ema_denoiser.state_dict(destination=current_sd, prefix="ema_denoiser.")

    pos_embed_load_policy = str(pos_embed_load_policy or "drop").lower()
    if pos_embed_load_policy not in {"drop", "interpolate"}:
        raise ValueError(
            "pos_embed_load_policy must be one of: drop, interpolate; "
            f"got {pos_embed_load_policy!r}"
        )

    for key in ["denoiser.pos_embed", "ema_denoiser.pos_embed"]:
        if key not in sd or key not in current_sd:
            continue
        if sd[key].shape == current_sd[key].shape:
            continue
        t = sd[key]
        cs = current_sd[key].shape
        if pos_embed_load_policy == "interpolate" and t.dim() == 3 and cs[0] == 1 and cs[2] == t.shape[2]:
            n0, n1 = t.shape[1], cs[1]
            s0, s1 = int(n0 ** 0.5), int(n1 ** 0.5)
            if s0 * s0 == n0 and s1 * s1 == n1:
                c = t.shape[2]
                w = t.reshape(1, s0, s0, c).permute(0, 3, 1, 2)
                w = torch.nn.functional.interpolate(w, (s1, s1), mode="bilinear", align_corners=False)
                sd[key] = w.permute(0, 2, 3, 1).reshape(1, n1, c).to(t.dtype)
                print(f"[Info] Interpolated '{key}' {s0}x{s0} -> {s1}x{s1}")
                continue
        print(f"[Warning] Shape mismatch '{key}', dropping. pos_embed_load_policy={pos_embed_load_policy}")
        del sd[key]

    for prefix, module in [("denoiser.", denoiser), ("ema_denoiser.", ema_denoiser),
                           ("diffusion_trainer.", diff_trainer)]:
        if module is None:
            continue
        sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if sub:
            module.load_state_dict(sub, strict=False)
    return ckpt


def make_predict_subdir(ckpt_path=None, loaded_step=None):
    if not ckpt_path:
        return "predict"

    stem = os.path.splitext(os.path.basename(str(ckpt_path)))[0]
    if stem and stem not in {"last", "latest"}:
        return f"predict_{stem}"

    if loaded_step is not None:
        try:
            step = int(loaded_step)
        except (TypeError, ValueError):
            step = None
        if step is not None and step > 0:
            return f"predict_step={step}"

    return "predict"


# ── eval loop ────────────────────────────────────────────────────────────────


def make_patch_coord_cell_grid(render_size, row_start, row_end, col_start, col_end, device, bs):
    render_h, render_w = render_size
    value_range = (
        (-1 + 2 * row_start / render_h, -1 + 2 * row_end / render_h),
        (-1 + 2 * col_start / render_w, -1 + 2 * col_end / render_w),
    )
    return make_coord_cell_grid(
        (row_end - row_start, col_end - col_start),
        value_range=value_range,
        device=device,
        bs=bs,
    )


def copy_grid_cache_attrs(dst, src):
    for attr in ("_cqar_grid_cache_key", "_cqar_grid_batch_mode"):
        if hasattr(src, attr):
            try:
                setattr(dst, attr, getattr(src, attr))
            except Exception:
                pass
    return dst


def contiguous_coord_cell(coord, cell):
    contig_coord = coord.contiguous()
    contig_cell = cell.contiguous()
    copy_grid_cache_attrs(contig_coord, coord)
    copy_grid_cache_attrs(contig_cell, cell)
    return contig_coord, contig_cell


def make_render_tile_specs(render_h, render_w, tile, pad):
    specs = []
    for row0 in range(0, render_h, tile):
        for col0 in range(0, render_w, tile):
            row_len = min(tile, render_h - row0)
            col_len = min(tile, render_w - col0)
            row_start = max(row0 - pad, 0)
            row_end = min(row0 + row_len + pad, render_h)
            col_start = max(col0 - pad, 0)
            col_end = min(col0 + col_len + pad, render_w)
            specs.append((
                row0,
                col0,
                row_len,
                col_len,
                row_start,
                row_end,
                col_start,
                col_end,
            ))
    return specs


def renderer_decode_input(ae, latents, render_size):
    return latents * ae.z_std + ae.z_mean if ae.use_ae_stats else latents


def decode_latents_parallel_tiles(ae, feat, render_size, tile_specs):
    render_h, render_w = render_size
    batch = feat.shape[0]
    output = None
    max_patch_h = max(spec[5] - spec[4] for spec in tile_specs)
    max_patch_w = max(spec[7] - spec[6] for spec in tile_specs)

    def _expand_interval(start, end, target, limit):
        extra = target - (end - start)
        before = extra // 2
        start = max(0, start - before)
        end = start + target
        if end > limit:
            end = limit
            start = end - target
        return start, end

    expanded_specs = []
    for spec in tile_specs:
        row_start, row_end = _expand_interval(spec[4], spec[5], max_patch_h, render_h)
        col_start, col_end = _expand_interval(spec[6], spec[7], max_patch_w, render_w)
        expanded_specs.append((
            spec[0],
            spec[1],
            spec[2],
            spec[3],
            row_start,
            row_end,
            col_start,
            col_end,
        ))
    coords = []
    cells = []
    for spec in expanded_specs:
        coord, cell = make_patch_coord_cell_grid(
            (render_h, render_w),
            spec[4],
            spec[5],
            spec[6],
            spec[7],
            device=feat.device,
            bs=None,
        )
        coords.append(coord)
        cells.append(cell)
    coord_keys = tuple(getattr(coord, "_cqar_grid_cache_key", None) for coord in coords)
    cell_keys = tuple(getattr(cell, "_cqar_grid_cache_key", None) for cell in cells)
    coord = torch.stack(coords, dim=0).contiguous()
    cell = torch.stack(cells, dim=0).contiguous()
    if coord_keys == cell_keys and all(key is not None for key in coord_keys):
        composite_key = ("stacked_tiles", coord_keys)
        try:
            coord._cqar_grid_cache_key = composite_key
            cell._cqar_grid_cache_key = composite_key
            coord._cqar_grid_batch_mode = 'batched'
            cell._cqar_grid_batch_mode = 'batched'
        except Exception:
            pass

    for image_idx in range(batch):
        image_feat = feat[image_idx:image_idx + 1]
        group_feat = image_feat.expand(len(expanded_specs), -1, -1, -1)
        patches = ae.renderer(group_feat, coord, cell)
        if output is None:
            output = torch.empty(
                batch,
                patches.shape[1],
                render_h,
                render_w,
                device=patches.device,
                dtype=patches.dtype,
            )
        for patch_idx, spec in enumerate(expanded_specs):
            row0, col0, row_len, col_len, row_start, _, col_start, _ = spec
            crop_top = row0 - row_start
            crop_left = col0 - col_start
            output[
                image_idx,
                :,
                row0:row0 + row_len,
                col0:col0 + col_len,
            ] = patches[
                patch_idx,
                :,
                crop_top:crop_top + row_len,
                crop_left:crop_left + col_len,
            ]
    if output is None:
        raise RuntimeError(f"Parallel patchwise decode produced no tiles for render size {render_size}.")
    return output


def decode_latents_patchwise(
    ae,
    latents,
    render_size,
    patch_size,
    patch_padding=0,
    parallel_tiles=False,
):
    if not hasattr(ae, "renderer"):
        if tuple(render_size) == tuple(latents.shape[-2:]):
            return ae.decode(latents)
        return ae.decode(latents, out_size=render_size)

    render_h, render_w = render_size
    if patch_size is None or patch_size <= 0 or (
        patch_size >= render_h and patch_size >= render_w
    ):
        return ae.decode(latents, out_size=render_size)

    feat = renderer_decode_input(ae, latents, render_size)
    batch = feat.shape[0]
    tile = int(patch_size)
    pad = int(patch_padding)
    tile_specs = make_render_tile_specs(render_h, render_w, tile, pad)
    if parallel_tiles:
        return decode_latents_parallel_tiles(ae, feat, render_size, tile_specs)

    output = None

    for spec in tile_specs:
        row0, col0, row_len, col_len, row_start, row_end, col_start, col_end = spec
        coord, cell = make_patch_coord_cell_grid(
            (render_h, render_w),
            row_start,
            row_end,
            col_start,
            col_end,
            device=feat.device,
            bs=batch,
        )
        coord, cell = contiguous_coord_cell(coord, cell)
        patch = ae.renderer(feat, coord, cell)
        if output is None:
            output = torch.empty(
                batch,
                patch.shape[1],
                render_h,
                render_w,
                device=patch.device,
                dtype=patch.dtype,
            )
        crop_top = row0 - row_start
        crop_left = col0 - col_start
        output[:, :, row0:row0 + row_len, col0:col0 + col_len] = patch[
            :,
            :,
            crop_top:crop_top + row_len,
            crop_left:crop_left + col_len,
        ]
    if output is None:
        raise RuntimeError(f"Patchwise decode produced no tiles for render size {render_size}.")
    return output


def decode_generated_latents(
    ae,
    latents,
    decode_size=None,
    overscan_pad=0,
    render_patch_size=None,
    render_patch_padding=0,
    render_parallel_tiles=False,
):
    if decode_size is None:
        target_size = (latents.shape[-2], latents.shape[-1])
    else:
        target_size = decode_size

    if overscan_pad <= 0:
        return decode_latents_patchwise(
            ae,
            latents,
            target_size,
            patch_size=render_patch_size,
            patch_padding=render_patch_padding,
            parallel_tiles=render_parallel_tiles,
        )

    target_h, target_w = target_size
    render_h = target_h + 2 * overscan_pad
    render_w = target_w + 2 * overscan_pad
    if render_h < target_h or render_w < target_w:
        raise ValueError(
            f"Overscan decode size {render_h}x{render_w} is smaller than target {target_h}x{target_w}"
        )

    decoded = decode_latents_patchwise(
        ae,
        latents,
        (render_h, render_w),
        patch_size=render_patch_size,
        patch_padding=render_patch_padding,
        parallel_tiles=render_parallel_tiles,
    )
    crop_top = (render_h - target_h) // 2
    crop_left = (render_w - target_w) // 2
    return decoded[
        :,
        :,
        crop_top:crop_top + target_h,
        crop_left:crop_left + target_w,
    ].contiguous()


def decode_generated_latents_per_sample(
    ae,
    latents,
    decode_sizes,
    overscan_pad=0,
    render_patch_size=None,
    render_patch_padding=0,
    render_parallel_tiles=False,
):
    decoded = []
    for idx, decode_size in enumerate(decode_sizes):
        decoded.append(
            decode_generated_latents(
                ae,
                latents[idx:idx + 1],
                decode_size=decode_size,
                overscan_pad=overscan_pad,
                render_patch_size=render_patch_size,
                render_patch_padding=render_patch_padding,
                render_parallel_tiles=render_parallel_tiles,
            )[0]
        )
    return decoded


def images_to_uint8(images):
    if isinstance(images, list):
        return [fp2uint8(image.unsqueeze(0))[0] for image in images]
    return fp2uint8(images)


def safe_path_component(value, fallback):
    value = fallback if value is None else str(value)
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)
    return safe or fallback


def intermediate_sample_dir(root_dir, metadata, batch_idx, sample_idx):
    item = metadata[sample_idx] if isinstance(metadata, (list, tuple)) else {}
    item = item if isinstance(item, dict) else {}
    filename = safe_path_component(
        item.get("filename", item.get("file_name", item.get("image_id", item.get("id")))),
        f"batch_{batch_idx:05d}_sample_{sample_idx:03d}",
    )
    seed = item.get("seed")
    if seed is not None:
        filename = f"{filename}_seed_{safe_path_component(seed, 'unknown')}"
    return os.path.join(root_dir, filename)


def save_latent_channel_mean_gray(latent, path):
    feature_map = latent.detach().float().mean(dim=0).cpu()
    min_value = feature_map.min()
    max_value = feature_map.max()
    denom = (max_value - min_value).clamp_min(1e-8)
    image = ((feature_map - min_value) / denom * 255.0).round().clamp(0, 255).to(torch.uint8).numpy()
    Image.fromarray(image).save(path)


def save_rendered_uint8_image(image, path):
    image = image.permute(1, 2, 0).detach().cpu().numpy()
    Image.fromarray(image).save(path)


def intermediate_step_name(step_idx, timesteps=None):
    if timesteps is None or step_idx >= len(timesteps):
        return f"step_{step_idx:04d}"
    timestep = timesteps[step_idx]
    if isinstance(timestep, torch.Tensor):
        timestep = timestep.detach().float().cpu().item()
    return f"step_{step_idx:04d}_t_{float(timestep):.6f}"


def save_intermediate_eval_steps(
    ae,
    x_trajs,
    metadata,
    batch_idx,
    target_dir,
    cfg,
    render_sizes=None,
    decode_size=None,
    decode_overscan_pad=0,
    render_patch_size=None,
    render_patch_padding=0,
    render_parallel_tiles=False,
    timesteps=None,
):
    if not cfg.get("enabled", False):
        return
    save_gray = cfg.get("save_latent_gray", True)
    save_rendered = cfg.get("save_rendered", True)
    if not save_gray and not save_rendered:
        return

    root = os.path.join(target_dir, cfg.get("subdir", "intermediates"))
    batch_size = x_trajs[-1].shape[0]
    sample_dirs = [intermediate_sample_dir(root, metadata, batch_idx, idx) for idx in range(batch_size)]

    for step_idx, latents in enumerate(x_trajs[1:], start=1):
        step_name = intermediate_step_name(step_idx, timesteps)
        if save_gray:
            for sample_idx in range(batch_size):
                step_dir = os.path.join(sample_dirs[sample_idx], step_name)
                os.makedirs(step_dir, exist_ok=True)
                save_latent_channel_mean_gray(
                    latents[sample_idx],
                    os.path.join(step_dir, "latent_mean.png"),
                )

        if save_rendered:
            if render_sizes is not None:
                decoded = decode_generated_latents_per_sample(
                    ae,
                    latents,
                    render_sizes,
                    overscan_pad=decode_overscan_pad,
                    render_patch_size=render_patch_size,
                    render_patch_padding=render_patch_padding,
                    render_parallel_tiles=render_parallel_tiles,
                )
            else:
                decoded = decode_generated_latents(
                    ae,
                    latents,
                    decode_size=decode_size,
                    overscan_pad=decode_overscan_pad,
                    render_patch_size=render_patch_size,
                    render_patch_padding=render_patch_padding,
                    render_parallel_tiles=render_parallel_tiles,
                )
            rendered = images_to_uint8(decoded)
            for sample_idx in range(batch_size):
                step_dir = os.path.join(sample_dirs[sample_idx], step_name)
                os.makedirs(step_dir, exist_ok=True)
                image = rendered[sample_idx] if isinstance(rendered, (list, tuple)) else rendered[sample_idx]
                save_rendered_uint8_image(image, os.path.join(step_dir, "render.png"))


def run_eval(mode, ae, conditioner, denoiser, ema_denoiser, sampler,
             dataset, batch_size, num_workers, save_hook,
             root_dir, step, rank, world_size, device, eval_original=False,
             decode_size=None, decode_overscan_pad=0, render_patch_size=None,
             render_patch_padding=0, render_parallel_tiles=False,
             inference_dtype=torch.bfloat16,
             intermediate_eval_cfg=None, predict_subdir=None):
    loader = make_eval_loader(dataset, batch_size, num_workers, rank, world_size)
    is_zero = rank == 0
    intermediate_eval_cfg = intermediate_eval_cfg or {"enabled": False}

    if mode == "predict":
        subdir = predict_subdir or "predict"
    elif decode_size is None:
        subdir = f"iter_{step}"
    elif decode_size[0] == decode_size[1]:
        subdir = f"iter_{step}_{decode_size[0]}"
    else:
        subdir = f"iter_{step}_{decode_size[0]}x{decode_size[1]}"
    save_root = save_hook.save_dir if save_hook is not None else mode
    target_dir = os.path.join(root_dir, save_root, subdir)

    if save_hook is not None:
        save_hook.save_start(target_dir, is_zero)
    if intermediate_eval_cfg.get("enabled", False):
        os.makedirs(os.path.join(target_dir, intermediate_eval_cfg.get("subdir", "intermediates")), exist_ok=True)
        if is_zero:
            print(
                "Save intermediate eval steps to "
                f"{os.path.join(target_dir, intermediate_eval_cfg.get('subdir', 'intermediates'))}"
            )
    if is_zero and decode_size is not None:
        print(f"[Info] Decode generated latents at size {decode_size[0]}x{decode_size[1]}")

    for idx, batch in enumerate(tqdm(loader, desc=mode.capitalize(), disable=not is_zero)):
        batch = move_to_device(batch, device)
        xT, y, metadata = batch
        with torch.inference_mode(), inference_autocast(device, inference_dtype):
            cond, uncond = conditioner(y)
            net = denoiser if eval_original else ema_denoiser
            render_sizes = metadata_render_sizes(metadata, xT.shape[0])
            x_trajs = None
            if intermediate_eval_cfg.get("enabled", False):
                samples, x_trajs = sampler_forward_with_trajectories(
                    sampler,
                    net,
                    xT,
                    cond,
                    uncond,
                )
            else:
                samples = sampler_forward(sampler, net, xT, cond, uncond)
            if render_sizes is not None:
                decoded = decode_generated_latents_per_sample(
                    ae,
                    samples,
                    render_sizes,
                    overscan_pad=decode_overscan_pad,
                    render_patch_size=render_patch_size,
                    render_patch_padding=render_patch_padding,
                    render_parallel_tiles=render_parallel_tiles,
                )
            else:
                decoded = decode_generated_latents(
                    ae,
                    samples,
                    decode_size=decode_size,
                    overscan_pad=decode_overscan_pad,
                    render_patch_size=render_patch_size,
                    render_patch_padding=render_patch_padding,
                    render_parallel_tiles=render_parallel_tiles,
                )
            if intermediate_eval_cfg.get("enabled", False):
                save_intermediate_eval_steps(
                    ae,
                    x_trajs,
                    metadata,
                    idx,
                    target_dir,
                    intermediate_eval_cfg,
                    render_sizes=render_sizes,
                    decode_size=decode_size,
                    decode_overscan_pad=decode_overscan_pad,
                    render_patch_size=render_patch_size,
                    render_patch_padding=render_patch_padding,
                    render_parallel_tiles=render_parallel_tiles,
                    timesteps=getattr(sampler, "timesteps", None),
                )
            samples = images_to_uint8(decoded)
        if save_hook is not None:
            save_hook.process_batch(samples, batch, is_zero, lambda t: all_gather(t, world_size))

    if save_hook is not None:
        save_hook.save_end()


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["fit", "validate", "predict"])
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values with OmegaConf dotlist entries, e.g. --set model.decode_size=512",
    )
    args = parser.parse_args()

    cfg_omegaconf = OmegaConf.load(args.config)
    if args.overrides:
        cfg_omegaconf = OmegaConf.merge(cfg_omegaconf, OmegaConf.from_dotlist(args.overrides))
    cfg = OmegaConf.to_container(cfg_omegaconf, resolve=True)
    tc = cfg.get("trainer", {})
    mc = cfg["model"]
    dc = cfg["data"]

    # env setup
    if cfg.get("torch_hub_dir"):
        os.environ["TORCH_HOME"] = cfg["torch_hub_dir"]
        torch.hub.set_dir(cfg["torch_hub_dir"])
    if cfg.get("huggingface_cache_dir"):
        os.environ["HUGGINGFACE_HUB_CACHE"] = cfg["huggingface_cache_dir"]
    seed_config = cfg.get("seed_everything", 0)
    seed = seed_config
    if seed is True:
        seed = random.randint(0, 2**32 - 1)
    if seed:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)

    # distributed
    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    is_zero = rank == 0

    # work dir
    root_dir = tc.get("default_root_dir", "./workdirs")
    tag = "".join(f"{k}_{v}" for k, v in cfg.get("tags", {}).items())
    root_dir = os.path.abspath(os.path.join(root_dir, tag))
    if is_zero:
        os.makedirs(root_dir, exist_ok=True)
        OmegaConf.save(
            cfg_omegaconf,
            os.path.join(root_dir, f"config-{args.command}-{time.strftime('%y%m%d%H%M')}.yaml"),
        )

    # ── build model (all classes directly imported, Cmd+Click to jump) ────
    ae_class_path = mc["ae"]["class_path"]
    if ae_class_path != "src.models.autoencoder.cqar_ae.CQARAutoencoder":
        raise ValueError("This release supports CQARAutoencoder only.")
    ae_init_args = dict(mc["ae"].get("init_args", {}))
    ae_ckpt_path = ae_init_args.pop("ae_ckpt", None) or mc.get("ae_ckpt")
    if ae_ckpt_path:
        ae = CQARAutoencoder.load_from_ae_checkpoint(ae_ckpt_path, **ae_init_args).to(device)
    else:
        ae = CQARAutoencoder(**ae_init_args).to(device)

    conditioner = Qwen3TextEncoder(**mc["conditioner"].get("init_args", {})).to(device)
    is_training = args.command == "fit"
    eval_original = parse_bool(mc.get("eval_original_model"), default=False)
    denoiser, ema_denoiser = build_denoisers(mc["denoiser"], device, is_training, eval_original)
    diff_trainer = None
    if is_training:
        from src.models.encoder import DINOv2
        from src.diffusion.flow_matching.training_rapd import RaPDTrainer

        dt_kw = dict(mc["diffusion_trainer"].get("init_args", {}))
        supported_trainers = {
            "src.diffusion.flow_matching.training_rapd.RaPDTrainer",
            "src.diffusion.flow_matching.training_repa_liif.REPATrainerLIIF",
        }
        if mc["diffusion_trainer"]["class_path"] not in supported_trainers:
            raise ValueError("This release supports RaPDTrainer only.")
        enc_cfg = dt_kw.pop("encoder")
        dt_kw["encoder"] = DINOv2(**enc_cfg.get("init_args", {}))
        if isinstance(dt_kw.get("scheduler"), str):
            dt_kw["scheduler"] = LinearScheduler()
        diff_trainer = RaPDTrainer(**dt_kw).to(device)

    ds_kw = dict(mc["diffusion_sampler"].get("init_args", {}))
    if isinstance(ds_kw.get("scheduler"), str):
        ds_kw["scheduler"] = LinearScheduler()
    if isinstance(ds_kw.get("w_scheduler"), str):
        ds_kw["w_scheduler"] = LinearScheduler()
    if isinstance(ds_kw.get("guidance_fn"), str):
        ds_kw["guidance_fn"] = simple_guidance_fn
    if isinstance(ds_kw.get("step_fn"), str):
        ds_kw["step_fn"] = ode_step_fn
    decode_size = parse_decode_size(mc.get("decode_size"))
    intermediate_eval_cfg = parse_intermediate_eval_cfg(mc.get("intermediate_eval"))
    inference_dtype = parse_inference_dtype(mc.get("inference_dtype", "bf16"))
    decode_overscan_pad = int(mc.get("decode_overscan_pad", mc.get("lr_reflect_pad", 0)))
    render_patch_size = mc.get("render_patch_size")
    if render_patch_size is not None:
        render_patch_size = int(render_patch_size)
    render_patch_padding = int(mc.get("render_patch_padding", 0))
    render_parallel_tiles = parse_bool(mc.get("render_parallel_tiles"), default=False)
    render_fast_blocks_cfg = mc.get("render_fast_blocks")
    if decode_overscan_pad < 0:
        raise ValueError("model.decode_overscan_pad must be non-negative")
    if render_patch_size is not None and render_patch_size <= 0:
        raise ValueError("model.render_patch_size must be positive when provided")
    if render_patch_padding < 0:
        raise ValueError("model.render_patch_padding must be non-negative")
    if is_zero and decode_size is not None:
        print(f"[Info] Eval decode size: {decode_size[0]}x{decode_size[1]}")
    if is_zero:
        print(f"[Info] Inference dtype: {inference_dtype_label(inference_dtype)}")
    if is_zero and decode_overscan_pad > 0:
        print(f"[Info] Eval decode overscan pad: {decode_overscan_pad}px")
    if is_zero and render_patch_size is not None:
        print(
            f"[Info] Eval patchwise decode: tile={render_patch_size}, "
            f"padding={render_patch_padding}"
        )
    if is_zero and render_parallel_tiles:
        if render_patch_size is None:
            print("[Warning] model.render_parallel_tiles has no effect without model.render_patch_size")
        else:
            print("[Info] Eval parallel tile decode enabled")
    if is_zero and intermediate_eval_cfg.get("enabled", False):
        print(
            "[Info] Intermediate eval saving enabled: "
            f"latent_gray={intermediate_eval_cfg['save_latent_gray']} "
            f"rendered={intermediate_eval_cfg['save_rendered']} "
            f"subdir={intermediate_eval_cfg['subdir']}"
        )
    if mc["diffusion_sampler"]["class_path"] != "src.diffusion.flow_matching.sampling.EulerSamplerJiT":
        raise ValueError("This release supports EulerSamplerJiT only.")
    diff_sampler = EulerSamplerJiT(**ds_kw).to(device)

    ema = None
    if is_training:
        ema_kw = mc["ema_tracker"].get("init_args", {}) if mc.get("ema_tracker") else {}
        ema = SimpleEMA(**ema_kw)

    no_grad(ae)
    no_grad(conditioner)
    for net in (denoiser, ema_denoiser):
        if net is not None:
            if not is_training or net is ema_denoiser:
                no_grad(net)
            net.compile()
    configure_renderer_fast_blocks(
        ae,
        render_fast_blocks_cfg,
        args.command,
        decode_size,
        is_zero,
    )
    if parse_bool(mc.get("compile_renderer"), default=False) and hasattr(ae, "renderer"):
        ae.renderer.compile()
        if is_zero:
            print("[Info] Compiled AE renderer")
            if bool(mc.get("ae", {}).get("init_args", {}).get("renderer_use_sdpa_attention", False)):
                print("[Info] Renderer SDPA is disabled inside torch.compile to avoid CUDA kernel launch limits")
    if ema_denoiser is not None:
        ema_denoiser.to(torch.float32)
    if is_training:
        ema.setup_models(denoiser, ema_denoiser)

    # callbacks
    cb_list = tc.get("callbacks", [])
    save_hook = SaveImagesHook(**cb_list[0].get("init_args", {})) if cb_list else None
    save_every = tc.get("save_every_n_steps", 100000)
    save_last = tc.get("save_last", True)

    # load checkpoint
    ckpt_path = args.ckpt_path or cfg.get("ckpt_path")
    finetune = parse_bool(cfg.get("finetune", cfg.get("fintune")), default=False)
    start_step = 0
    ckpt = None
    loaded_step = None
    if ckpt_path:
        pos_embed_load_policy = cfg.get("pos_embed_load_policy", mc.get("pos_embed_load_policy", "drop"))
        ckpt = load_ckpt(
            ckpt_path,
            denoiser,
            ema_denoiser,
            diff_trainer,
            device if is_training else "cpu",
            pos_embed_load_policy=pos_embed_load_policy,
        )
        loaded_step = ckpt.get("global_step", 0)
        reset_step_on_load = parse_bool(cfg.get("reset_step_on_load"), default=False)
        start_step = 0 if reset_step_on_load else loaded_step
        if is_zero:
            print(f"[Info] Loaded checkpoint from {ckpt_path}  step={loaded_step}")
            if reset_step_on_load:
                print("[Info] reset_step_on_load=true: starting optimizer schedule at step=0")
    predict_subdir = tc.get("predict_subdir") or make_predict_subdir(ckpt_path, loaded_step)
    if args.command == "predict" and is_zero:
        print(f"[Info] Predict images will be saved under '{predict_subdir}'")

    # ── predict / validate ───────────────────────────────────────────────
    if args.command in ("predict", "validate"):
        # Release unused training weights and optimizer state loaded from the checkpoint.
        del ckpt
        ds_key = "pred_dataset" if args.command == "predict" else "eval_dataset"
        ds_cfg = copy.deepcopy(dc[ds_key])
        ds_class_path = str(ds_cfg.get("class_path", ""))
        ds_init_args = ds_cfg.get("init_args")
        if ds_init_args is None:
            ds_init_args = {}
            ds_cfg["init_args"] = ds_init_args
        if (
            args.command == "predict"
            and "dpg" in ds_class_path.lower()
            and not isinstance(seed_config, bool)
            and seed_config is not None
            and ds_init_args.get("seed") is None
            and ds_init_args.get("seeds") is None
        ):
            ds_init_args["seed"] = int(seed_config)
            if is_zero:
                print(f"[Info] Using seed_everything={seed_config} as DPGDataset seed")
        dataset = instantiate_from_config(ds_cfg, "src.data.dataset.geneval.GenEvalDataset")
        bs = dc.get("pred_batch_size", dc.get("eval_batch_size", 32))
        nw = dc.get("pred_num_workers", dc.get("eval_num_workers", 4))
        run_eval(args.command, ae, conditioner, denoiser, ema_denoiser, diff_sampler,
                 dataset, bs, nw, save_hook, root_dir, start_step,
                 rank, world_size, device, eval_original,
                 decode_size, decode_overscan_pad, render_patch_size,
                 render_patch_padding, render_parallel_tiles, inference_dtype,
                 intermediate_eval_cfg, predict_subdir)
        cleanup_distributed(world_size)
        return

    # ── fit ───────────────────────────────────────────────────────────────
    # optimizer
    opt_kw = dict(mc["optimizer"].get("init_args", {}))
    denoiser_params = filter_nograd_tensors(denoiser.parameters())
    diff_trainer_params = filter_nograd_tensors(diff_trainer.parameters())
    params = []
    if denoiser_params:
        params.append({"params": denoiser_params})
    if diff_trainer_params:
        params.append({"params": diff_trainer_params})
    if not params:
        raise ValueError("No trainable parameters found for optimizer.")
    if is_zero:
        print(
            f"[Info] Trainable params: denoiser={count_trainable_parameters(denoiser):,} "
            f"diff_trainer={count_trainable_parameters(diff_trainer):,}"
        )
    optimizer = AdamW(params, **opt_kw)
    lr_sched = None
    if mc.get("lr_scheduler"):
        lr_kw = mc["lr_scheduler"].get("init_args", {})
        lr_sched = ConstantWithWarmup(**lr_kw)(optimizer)

    if ckpt is not None:
        if finetune:
            if is_zero:
                print("[Info] Finetune mode: using optimizer/lr from config, not checkpoint")
        else:
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if lr_sched and ckpt.get("lr_scheduler"):
                lr_sched.load_state_dict(ckpt["lr_scheduler"])
            if is_zero and "optimizer" in ckpt:
                print("[Info] Resumed optimizer/lr from checkpoint")
        if is_zero:
            lrs = sorted({group["lr"] for group in optimizer.param_groups})
            print(f"[Info] Optimizer lr after checkpoint handling: {lrs}")
        del ckpt

    # dataloader
    tr_cfg = dc["train_dataset"]
    train_ds = instantiate_from_config(tr_cfg)
    train_loader = make_train_loader(
        train_ds, dc.get("train_batch_size", 32), dc.get("train_num_workers", 4),
        rank, world_size, dc.get("train_prefetch_factor", 8),
    )
    if is_zero:
        ds_len = len(train_ds) if hasattr(train_ds, "__len__") else "N/A (iterable)"
        print(f"[Data] {type(train_ds).__name__} len={ds_len}")

    # DDP
    class _TrainModules(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.denoiser = denoiser
            self.diff_trainer = diff_trainer

        def forward(self, x, ema_dn, cond, uncond, meta):
            return self.diff_trainer(self.denoiser, ema_dn, diff_sampler,
                                     x, cond, uncond, meta)

    train_mod = _TrainModules()
    ddp = DDP(train_mod, device_ids=[local_rank], find_unused_parameters=True) if world_size > 1 else train_mod

    # TensorBoard
    writer = None
    if is_zero:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(root_dir, "tb_logs"))

    # training config
    max_steps = tc.get("max_steps", 800000)
    accum = tc.get("accumulate_grad_batches", 1)
    clip_val = tc.get("gradient_clip_val")
    log_interval = tc.get("log_every_n_steps", 50)
    val_interval = tc.get("val_check_interval", 0)
    validate_at_start = parse_bool(tc.get("validate_at_start"), default=False)

    eval_ds = instantiate_from_config(dc["eval_dataset"], "src.data.dataset.geneval.GenEvalDataset") if dc.get("eval_dataset") else None
    eval_bs = dc.get("eval_batch_size", 32)
    eval_nw = dc.get("eval_num_workers", 4)

    # ── training loop ────────────────────────────────────────────────────
    step = start_step
    if validate_at_start and eval_ds:
        ema.synchronize()
        if is_zero:
            print(f"[Info] validate_at_start=true: running validation before training at step={step}")
        run_eval("validate", ae, conditioner, denoiser, ema_denoiser, diff_sampler,
                 eval_ds, eval_bs, eval_nw, save_hook, root_dir, step,
                 rank, world_size, device, eval_original,
                 decode_size, decode_overscan_pad, render_patch_size,
                 render_patch_padding, render_parallel_tiles, inference_dtype,
                 intermediate_eval_cfg)
        if world_size > 1:
            dist.barrier()

    pbar = tqdm(total=max_steps, initial=step, desc="Training", disable=not is_zero)
    ddp.train()
    optimizer.zero_grad()

    micro_step = 0
    epoch = 0
    while step < max_steps:
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(train_loader):
            batch = move_to_device(batch, device)
            x, y, metadata = batch
            if metadata is None:
                metadata = {}
            metadata["global_step"] = step
            micro_step += 1
            last_accum = micro_step % accum == 0

            ctx = ddp.no_sync() if world_size > 1 and not last_accum else nullcontext()
            with ctx:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    x_enc = ae.encode(x)
                    cond, uncond = conditioner(y, metadata)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss_dict = ddp(x_enc, ema_denoiser, cond, uncond, metadata)
                    (loss_dict["loss"] / accum).backward()

            if not last_accum:
                continue

            # optimizer step
            if clip_val:
                torch.nn.utils.clip_grad_norm_(train_mod.parameters(), clip_val)
            optimizer.step()
            optimizer.zero_grad()
            if lr_sched:
                lr_sched.step()
            step += 1
            ema.step(step)

            # log — only sync GPU→CPU every log_interval steps to avoid stalling GPU 0
            if step % log_interval == 0 and is_zero:
                scalar_dict = {
                    k: v.item() if isinstance(v, torch.Tensor) else v
                    for k, v in loss_dict.items()
                }
                if writer:
                    for k, sv in scalar_dict.items():
                        writer.add_scalar(f"train/{k}", sv, step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
                    if len(optimizer.param_groups) > 1:
                        for group_idx, group in enumerate(optimizer.param_groups):
                            writer.add_scalar(f"train/lr_group_{group_idx}", group["lr"], step)
                pbar.set_postfix({k: f"{sv:.4f}" for k, sv in scalar_dict.items()})

            # checkpoint
            # if step % 5000 == 0 and is_zero:
            #     save_ckpt(os.path.join(root_dir, "latest.ckpt"),
            #               denoiser, ema_denoiser, diff_trainer, optimizer, lr_sched, step)
            if step % save_every == 0:
                ema.synchronize()
                if is_zero:
                    save_ckpt(os.path.join(root_dir, f"epoch=0-step={step}.ckpt"),
                              denoiser, ema_denoiser, diff_trainer, optimizer, lr_sched, step)
                    if save_last:
                        save_ckpt(os.path.join(root_dir, "last.ckpt"),
                                  denoiser, ema_denoiser, diff_trainer, optimizer, lr_sched, step)
                if world_size > 1:
                    dist.barrier()

            # validation
            if val_interval and step % val_interval == 0 and eval_ds:
                ema.synchronize()
                run_eval("validate", ae, conditioner, denoiser, ema_denoiser, diff_sampler,
                         eval_ds, eval_bs, eval_nw, save_hook, root_dir, step,
                         rank, world_size, device, eval_original,
                         decode_size, decode_overscan_pad, render_patch_size,
                         render_patch_padding, render_parallel_tiles, inference_dtype,
                         intermediate_eval_cfg)
                if world_size > 1:
                    dist.barrier()
                ddp.train()

            # progress
            pbar.update(1)

            if step >= max_steps:
                break
        epoch += 1
        if step >= max_steps:
            break

    # final save
    ema.synchronize()
    if is_zero:
        save_ckpt(os.path.join(root_dir, "last.ckpt"),
                  denoiser, ema_denoiser, diff_trainer, optimizer, lr_sched, step)
        print(f"[Done] step={step}  checkpoint: {root_dir}/last.ckpt")
    if world_size > 1:
        dist.barrier()

    pbar.close()
    if writer:
        writer.close()
    cleanup_distributed(world_size)


if __name__ == "__main__":
    main()
