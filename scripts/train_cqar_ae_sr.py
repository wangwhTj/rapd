#!/usr/bin/env python3
"""Train CQAR AE with dense patch supervision for LIIF-style SR inputs.

This script keeps the LIIF-SR style input construction:
1. sample an HR crop at a random scale
2. downsample it to a fixed LR input

But supervision is changed to match CQAR / INFD-style rendering:
1. choose a fixed-size dense HR patch from the HR crop
2. build a dense coord/cell grid for that patch
3. supervise the rendered patch with L1 + LPIPS
4. align latent features with a frozen DINOv2 teacher
"""

import argparse
import glob
import io
import os
import random
from contextlib import nullcontext
from pathlib import Path

from PIL import Image

import lpips
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, Dataset, DistributedSampler, IterableDataset
from torchvision import transforms
from tqdm.auto import tqdm
import webdataset as wds

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cqar_training_utils as training_utils

from src.models.autoencoder.cqar_ae import CQARAutoencoder
from src.models.autoencoder.coordinates import make_coord_cell_grid

Image.MAX_IMAGE_PIXELS = None


def build_vf_teacher(args) -> nn.Module:
    teacher = training_utils.FrozenDINOv2Teacher(
        weight_path=args.vf_dino_weight_path,
        repo_dir=args.vf_dino_repo_dir,
        base_patch_size=args.vf_base_patch_size,
    )
    teacher.teacher_type = 'dino'
    teacher.teacher_name = 'dinov2_vitb14'
    return teacher


def build_cqar_sample(
    img: Image.Image,
    inp_size: int,
    patch_size: int,
    scale_min: float,
    scale_max: float,
    augment: bool,
    min_size: int = 0,
    include_hr: bool = True,
    return_reason: bool = False,
):
    """Create one CQAR-compatible LR / dense-HR training sample from a PIL image."""
    if not isinstance(img, Image.Image):
        return (None, 'not_image') if return_reason else None

    img = img.convert('RGB')
    if min_size > 0 and min(img.size) < min_size:
        return (None, 'below_min_size') if return_reason else None

    img_hr = transforms.ToTensor()(img)
    scale = random.uniform(scale_min, scale_max)
    hr_size = round(inp_size * scale)

    if img_hr.shape[-2] < hr_size or img_hr.shape[-1] < hr_size:
        return (None, 'too_small_for_scale') if return_reason else None

    x0 = random.randint(0, img_hr.shape[-2] - hr_size)
    y0 = random.randint(0, img_hr.shape[-1] - hr_size)
    crop_hr = img_hr[:, x0:x0 + hr_size, y0:y0 + hr_size].contiguous()
    crop_lr = training_utils.resize_tensor(crop_hr, inp_size)

    if augment:
        crop_lr, crop_hr = training_utils.augment_pair(crop_lr, crop_hr)

    if patch_size > hr_size:
        return (None, 'patch_larger_than_hr') if return_reason else None

    px0 = random.randint(0, hr_size - patch_size)
    py0 = random.randint(0, hr_size - patch_size)
    gt_patch = crop_hr[:, px0:px0 + patch_size, py0:py0 + patch_size].contiguous()

    range_x = [2 * (px0 / hr_size) - 1, 2 * ((px0 + patch_size) / hr_size) - 1]
    range_y = [2 * (py0 / hr_size) - 1, 2 * ((py0 + patch_size) / hr_size) - 1]
    coord, cell = make_coord_cell_grid(
        (patch_size, patch_size),
        value_range=[range_x, range_y],
    )

    sample = {
        'inp': crop_lr.mul(2.0).sub(1.0),
        'coord': coord,
        'cell': cell,
        'gt': gt_patch.mul(2.0).sub(1.0),
        'scale': torch.tensor(scale, dtype=torch.float32),
    }
    if include_hr:
        sample['hr'] = crop_hr.mul(2.0).sub(1.0)
    return (sample, None) if return_reason else sample


class SRImplicitImageFolderCQAR(Dataset):
    def __init__(
        self,
        root_dir,
        inp_size=256,
        patch_size=256,
        scale_min=1.0,
        scale_max=4.0,
        augment=True,
        min_size=0,
        include_hr=True,
        exts='jpg,jpeg,png,webp',
    ):
        self.root = Path(root_dir)
        ext_list = [e.strip().lower().lstrip('.') for e in exts.split(',') if e.strip()]
        files = []
        for ext in ext_list:
            files.extend(self.root.rglob(f'*.{ext}'))
            files.extend(self.root.rglob(f'*.{ext.upper()}'))
        self.files = sorted(set(files))
        if not self.files:
            raise ValueError(f'No images found under: {root_dir}')

        self.inp_size = inp_size
        self.patch_size = patch_size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.min_size = min_size
        self.include_hr = include_hr

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        for attempt in range(8):
            path = self.files[(idx + attempt) % len(self.files)]
            try:
                img = Image.open(path).convert('RGB')
            except Exception:
                continue

            sample = build_cqar_sample(
                img=img,
                inp_size=self.inp_size,
                patch_size=self.patch_size,
                scale_min=self.scale_min,
                scale_max=self.scale_max,
                augment=self.augment,
                min_size=self.min_size,
                include_hr=self.include_hr,
            )
            if sample is not None:
                return sample

        raise RuntimeError(f'Failed to create a valid sample near index {idx}')


class SRImplicitWebDatasetCQAR(IterableDataset):
    def __init__(
        self,
        urls,
        inp_size=256,
        patch_size=256,
        scale_min=1.0,
        scale_max=4.0,
        augment=True,
        min_size=0,
        shuffle_buffer=1000,
        include_hr=True,
    ):
        super().__init__()
        self.urls = list(urls)
        self.inp_size = inp_size
        self.patch_size = patch_size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.min_size = min_size
        self.shuffle_buffer = shuffle_buffer
        self.include_hr = include_hr

    def _extract_image(self, sample):
        for key in ['jpg', 'jpeg', 'png', 'webp', 'image', 'img', 'output_image', 'input_image', 'data']:
            if key in sample:
                value = sample[key]
                if isinstance(value, Image.Image):
                    return value
                if isinstance(value, (bytes, bytearray)):
                    try:
                        return Image.open(io.BytesIO(value))
                    except Exception:
                        pass
        for value in sample.values():
            if isinstance(value, Image.Image):
                return value
            if isinstance(value, (bytes, bytearray)):
                try:
                    return Image.open(io.BytesIO(value))
                except Exception:
                    continue
        return None

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        dataset = wds.WebDataset(
            self.urls,
            shardshuffle=100 if self.shuffle_buffer > 0 else 0,
            nodesplitter=wds.split_by_node,
        )
        dataset = dataset.compose(wds.split_by_worker)
        dataset = dataset.decode('pil')
        if self.shuffle_buffer > 0:
            dataset = dataset.shuffle(self.shuffle_buffer)
        dataset = dataset.repeat()

        seen = 0
        yielded = 0
        reason_counts = {}
        first_missing_keys = None

        for sample in dataset:
            try:
                seen += 1
                img = self._extract_image(sample)
                if img is None:
                    reason_counts['missing_image_key'] = reason_counts.get('missing_image_key', 0) + 1
                    if first_missing_keys is None:
                        first_missing_keys = sorted(str(k) for k in sample.keys())
                    continue
                out, reason = build_cqar_sample(
                    img=img,
                    inp_size=self.inp_size,
                    patch_size=self.patch_size,
                    scale_min=self.scale_min,
                    scale_max=self.scale_max,
                    augment=self.augment,
                    min_size=self.min_size,
                    include_hr=self.include_hr,
                    return_reason=True,
                )
                if out is not None:
                    yielded += 1
                    yield out
                    continue
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                if yielded == 0 and seen >= 2048:
                    detail = ', '.join(f'{k}={v}' for k, v in sorted(reason_counts.items()))
                    extra = ''
                    if first_missing_keys is not None:
                        extra = f'; first_missing_sample_keys={first_missing_keys}'
                    raise RuntimeError(
                        'WebDataset yielded no valid CQAR SR samples after '
                        f'{seen} records on worker {worker_id}. '
                        f'Reasons: {detail}{extra}'
                    )
            except GeneratorExit:
                raise
            except RuntimeError:
                raise
            except Exception:
                continue

    def probe(self, max_records=128):
        dataset = wds.WebDataset(
            self.urls,
            shardshuffle=0,
            nodesplitter=wds.split_by_node,
        ).decode('pil')

        seen = 0
        valid = 0
        reason_counts = {}
        example_sizes = []
        first_missing_keys = None

        for sample in dataset:
            seen += 1
            img = self._extract_image(sample)
            if img is None:
                reason_counts['missing_image_key'] = reason_counts.get('missing_image_key', 0) + 1
                if first_missing_keys is None:
                    first_missing_keys = sorted(str(k) for k in sample.keys())
            else:
                if len(example_sizes) < 8 and isinstance(img, Image.Image):
                    example_sizes.append(tuple(img.size))
                _, reason = build_cqar_sample(
                    img=img,
                    inp_size=self.inp_size,
                    patch_size=self.patch_size,
                    scale_min=self.scale_min,
                    scale_max=self.scale_max,
                    augment=self.augment,
                    min_size=self.min_size,
                    include_hr=self.include_hr,
                    return_reason=True,
                )
                if reason is None:
                    valid += 1
                else:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if seen >= max_records:
                break

        return {
            'seen': seen,
            'valid': valid,
            'reason_counts': reason_counts,
            'example_sizes': example_sizes,
            'first_missing_keys': first_missing_keys,
        }


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def build_train_loader(args, use_ddp, include_hr):
    train_dirs = args.train_dir if isinstance(args.train_dir, (list, tuple)) else [args.train_dir]
    tar_files = []
    for train_dir in train_dirs:
        tar_files.extend(glob.glob(os.path.join(train_dir, '**/*.tar'), recursive=True))
    tar_files = sorted(set(tar_files))

    def collate_fn(samples):
        batch = {
            'inp': torch.stack([sample['inp'] for sample in samples], dim=0),
            'coord': torch.stack([sample['coord'] for sample in samples], dim=0),
            'cell': torch.stack([sample['cell'] for sample in samples], dim=0),
            'gt': torch.stack([sample['gt'] for sample in samples], dim=0),
            'scale': torch.stack([sample['scale'] for sample in samples], dim=0),
        }
        if include_hr:
            batch['hr'] = [sample['hr'] for sample in samples]
        return batch

    if tar_files:
        if is_main():
            print(f'Using WebDataset format: {len(tar_files)} tar files found')
        train_ds = SRImplicitWebDatasetCQAR(
            urls=tar_files,
            inp_size=args.inp_size,
            patch_size=args.patch_size,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            augment=args.augment,
            min_size=args.min_size,
            shuffle_buffer=args.shuffle_buffer,
            include_hr=include_hr,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        sampler = None
        is_webdataset = True
    else:
        datasets = [
            SRImplicitImageFolderCQAR(
                root_dir=train_dir,
                inp_size=args.inp_size,
                patch_size=args.patch_size,
                scale_min=args.scale_min,
                scale_max=args.scale_max,
                augment=args.augment,
                min_size=args.min_size,
                include_hr=include_hr,
                exts=args.exts,
            )
            for train_dir in train_dirs
        ]
        train_ds = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
        sampler = DistributedSampler(train_ds, shuffle=True) if use_ddp else None
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        if is_main():
            print(f'Using image-folder format: {len(train_ds)} images from {len(train_dirs)} dirs')
        is_webdataset = False

    return train_loader, sampler, train_ds, is_webdataset


def train(args):
    use_ddp = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    local_rank = 0
    if use_ddp:
        dist.init_process_group('nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_vf = args.vf_loss_weight > 0

    if is_main():
        os.makedirs(args.save_dir, exist_ok=True)
        print(f'Device: {device}, DDP: {use_ddp}')
        if use_ddp:
            print(f'World size: {dist.get_world_size()}')
        print(
            f'CQAR SR training: inp_size={args.inp_size}, patch_size={args.patch_size}, '
            f'scale=[{args.scale_min}, {args.scale_max}]'
        )
        print(f'VF enabled: {use_vf}')
        print('Building training loader...')

    train_loader, sampler, train_ds, is_webdataset = build_train_loader(args, use_ddp, include_hr=use_vf)
    if is_main() and is_webdataset and args.webdataset_probe_records > 0:
        print(f'Probing WebDataset with {args.webdataset_probe_records} records...')
        probe = train_ds.probe(max_records=args.webdataset_probe_records)
        print(
            f'WebDataset probe: seen={probe["seen"]}, valid={probe["valid"]}, '
            f'reasons={probe["reason_counts"]}'
        )
        if probe['example_sizes']:
            print(f'Example decoded image sizes: {probe["example_sizes"]}')
        if probe['first_missing_keys'] is not None:
            print(f'First sample without image had keys: {probe["first_missing_keys"]}')
        if probe['valid'] == 0:
            raise RuntimeError(
                'WebDataset probe found zero valid samples. '
                'Adjust image key handling, scale range, or input data.'
            )

    model = CQARAutoencoder(
        feat_channels=args.feat_channels,
        hidden=args.hidden,
        use_ae_stats=False,
        renderer_block_type=args.renderer_block_type,
        renderer_num_blocks=args.renderer_num_blocks,
        renderer_num_heads=args.renderer_num_heads,
        renderer_ffn_expansion_factor=args.renderer_ffn_expansion_factor,
        renderer_attention_bias=args.renderer_attention_bias,
        renderer_layer_norm_bias=args.renderer_layer_norm_bias,
        renderer_window_size=args.renderer_window_size,
    ).to(device)
    if is_main():
        print('Model initialized.')

    teacher = None
    vf_proj = None
    if use_vf:
        teacher = build_vf_teacher(args)
        teacher = training_utils.maybe_cast_teacher(teacher, device, args.vf_teacher_dtype)
        teacher.eval()
        if is_main():
            print(
                f'VF teacher: type={getattr(teacher, "teacher_type", args.vf_teacher_type)}, '
                f'name={getattr(teacher, "teacher_name", "unknown")}, '
                f'feature_dim={teacher.feature_dim}'
            )

        vf_proj_in_dim = (
            args.feat_channels if args.vf_proj_direction == 'latent_to_teacher'
            else teacher.feature_dim
        )
        vf_proj_out_dim = (
            teacher.feature_dim if args.vf_proj_direction == 'latent_to_teacher'
            else args.feat_channels
        )
        vf_proj = nn.Conv2d(vf_proj_in_dim, vf_proj_out_dim, kernel_size=1, bias=True).to(device)


    if use_ddp:
        model = DDP(model, device_ids=[local_rank])
        if use_vf:
            vf_proj = DDP(vf_proj, device_ids=[local_rank])
    raw_model = model.module if use_ddp else model
    raw_vf_proj = vf_proj.module if (use_ddp and use_vf) else vf_proj

    if is_main():
        print('Initializing LPIPS...')
    lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)
    if is_main():
        print('LPIPS initialized.')

    optim_params = list(raw_model.parameters())
    if use_vf:
        optim_params.extend(list(raw_vf_proj.parameters()))
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_steps,
        eta_min=args.min_lr,
    )
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    l1_fn = nn.L1Loss()

    step = 0
    epoch = 0
    best_loss = float('inf')
    running_loss = running_l1 = running_lpips = running_scale = 0.0
    running_mcos = running_mdms = running_vf = running_vf_weight = 0.0
    log_count = 0

    if args.resume and os.path.isfile(args.resume):
        if is_main():
            print(f'Resuming from: {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        raw_model.encoder.load_state_dict(ckpt['encoder'])
        raw_model.renderer.load_state_dict(ckpt['renderer'])
        if use_vf and 'vf_proj' in ckpt:
            raw_vf_proj.load_state_dict(ckpt['vf_proj'])
        try:
            resume_has_vf = 'vf_proj' in ckpt
            if use_vf == resume_has_vf:
                optimizer.load_state_dict(ckpt['optimizer'])
                scheduler.load_state_dict(ckpt['scheduler'])
            elif is_main():
                print('  Resume checkpoint VF configuration differs from current args; reinitializing optimizer/scheduler.')
        except (KeyError, ValueError):
            if is_main():
                print('  Resume optimizer/scheduler state is incompatible; reinitializing them.')
        step = ckpt.get('step', 0)
        epoch = ckpt.get('epoch', 0)
        best_loss = ckpt.get('best_loss', float('inf'))
        if is_main():
            print(f'  Resumed at step={step}, epoch={epoch}, best_loss={best_loss:.4f}')
            if use_vf and 'vf_proj' not in ckpt:
                print('  Resume checkpoint has no vf_proj; starting VF projection from scratch.')

    if is_main():
        pbar = tqdm(total=args.max_steps, desc='Training')
        if step > 0:
            pbar.update(step)

    seen_first_batch = False
    while step < args.max_steps:
        epoch += 1
        if sampler is not None:
            sampler.set_epoch(epoch)

        for batch in train_loader:
            if step >= args.max_steps:
                break

            if is_main() and not seen_first_batch:
                print('First batch fetched.')
                seen_first_batch = True

            inp = batch['inp'].to(device, non_blocking=True)
            coord = batch['coord'].to(device, non_blocking=True)
            cell = batch['cell'].to(device, non_blocking=True)
            gt = batch['gt'].to(device, non_blocking=True)
            sample_scale = batch['scale'].to(device, non_blocking=True)
            hr_list = None
            if use_vf:
                hr_list = [hr.to(device, non_blocking=True) for hr in batch['hr']]


            with (
                torch.amp.autocast('cuda', dtype=torch.bfloat16)
                if torch.cuda.is_available()
                else nullcontext()
            ):
                pred, z = model(
                    inp,
                    coord=coord,
                    cell=cell,
                    return_latent=True,
                )
                l1_loss = l1_fn(pred, gt)
                lp_loss = lpips_fn(pred, gt).mean()
                rec_loss = l1_loss + args.lpips_weight * lp_loss
                loss = rec_loss

                if use_vf:
                    mcos_terms = []
                    mdms_terms = []
                    vf_terms = []
                    cos_terms = []
                    for sample_idx, hr in enumerate(hr_list):
                        with torch.no_grad():
                            teacher_feat = teacher(hr.unsqueeze(0))
                        mcos_loss_i, mdms_loss_i, vf_loss_i, vf_cos_i = training_utils.compute_vf_losses(
                            latent=z[sample_idx:sample_idx + 1],
                            teacher_feat=teacher_feat,
                            vf_proj=vf_proj,
                            proj_direction=args.vf_proj_direction,
                            cos_margin=args.vf_cos_margin,
                            distmat_margin=args.vf_distmat_margin,
                            cos_weight=args.vf_cos_weight,
                            distmat_weight=args.vf_distmat_weight,
                            max_tokens=args.vf_max_tokens,
                        )
                        mcos_terms.append(mcos_loss_i)
                        mdms_terms.append(mdms_loss_i)
                        vf_terms.append(vf_loss_i)
                        cos_terms.append(vf_cos_i)

                    mcos_loss = torch.stack(mcos_terms).mean()
                    mdms_loss = torch.stack(mdms_terms).mean()
                    vf_loss = torch.stack(vf_terms).mean()
                    vf_cos = torch.stack(cos_terms).mean()

                    if args.vf_adaptive_weight:
                        try:
                            vf_weight = training_utils.calculate_adaptive_vf_weight(
                                rec_loss=rec_loss,
                                vf_loss=vf_loss,
                                last_layer=raw_model.encoder.tail.weight,
                                base_weight=args.vf_loss_weight,
                            )
                        except RuntimeError:
                            if step == 0 and is_main():
                                print('Warning: adaptive VF weight failed once, falling back to 0 for this step.')
                            vf_weight = torch.tensor(0.0, device=device)
                    else:
                        vf_weight = torch.tensor(args.vf_loss_weight, device=device)

                    loss = loss + vf_weight * vf_loss


            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(optim_params, args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(optim_params, args.grad_clip_norm)
                optimizer.step()
            scheduler.step()


            step += 1
            batch_size = inp.shape[0]
            running_loss += loss.item() * batch_size
            running_l1 += l1_loss.item() * batch_size
            running_lpips += lp_loss.item() * batch_size
            running_scale += sample_scale.mean().item() * batch_size
            if use_vf:
                running_mcos += mcos_loss.item() * batch_size
                running_mdms += mdms_loss.item() * batch_size
                running_vf += vf_loss.item() * batch_size
                running_vf_weight += vf_weight.item() * batch_size
            log_count += batch_size
            current_loss = loss.item()

            if is_main():
                postfix = {
                    'loss': f'{loss.item():.4f}',
                    'l1': f'{l1_loss.item():.4f}',
                    'lpips': f'{lp_loss.item():.4f}',
                    'scale': f'{sample_scale.mean().item():.2f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                }
                if use_vf:
                    postfix['vf'] = f'{vf_loss.item():.4f}'
                    postfix['vfw'] = f'{vf_weight.item():.4f}'
                    postfix['cos'] = f'{vf_cos.item():.4f}'
                pbar.update(1)
                pbar.set_postfix(**postfix)

            if is_main() and step % args.log_every == 0:
                avg_loss = running_loss / max(log_count, 1)
                avg_l1 = running_l1 / max(log_count, 1)
                avg_lpips = running_lpips / max(log_count, 1)
                avg_scale = running_scale / max(log_count, 1)
                message = (
                    f'[Step {step}] loss={avg_loss:.4f}  l1={avg_l1:.4f}  '
                    f'lpips={avg_lpips:.4f}  scale={avg_scale:.2f}'
                )
                if use_vf:
                    avg_mcos = running_mcos / max(log_count, 1)
                    avg_mdms = running_mdms / max(log_count, 1)
                    avg_vf = running_vf / max(log_count, 1)
                    avg_vf_weight = running_vf_weight / max(log_count, 1)
                    message += (
                        f'  mcos={avg_mcos:.4f}  mdms={avg_mdms:.4f}  '
                        f'vf={avg_vf:.4f}  vfw={avg_vf_weight:.4f}'
                    )
                message += f'  lr={scheduler.get_last_lr()[0]:.2e}'
                tqdm.write(message)
                running_loss = running_l1 = running_lpips = running_scale = 0.0
                running_mcos = running_mdms = running_vf = running_vf_weight = 0.0
                log_count = 0

            if is_main() and (step % args.save_every == 0 or step == args.max_steps):
                if current_loss < best_loss:
                    best_loss = current_loss
                    tqdm.write(f'  -> best checkpoint (loss={best_loss:.4f})')

                ckpt = {
                    'step': step,
                    'epoch': epoch,
                    'best_loss': best_loss,
                    'encoder': raw_model.encoder.state_dict(),
                    'renderer': raw_model.renderer.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'args': vars(args),
                }
                if use_vf:
                    ckpt['vf_proj'] = raw_vf_proj.state_dict()
                    ckpt['vf_meta'] = {
                        'teacher_type': getattr(teacher, 'teacher_type', args.vf_teacher_type),
                        'teacher_name': getattr(teacher, 'teacher_name', None),
                        'teacher_feature_dim': teacher.feature_dim,
                        'proj_direction': args.vf_proj_direction,
                        'cos_margin': args.vf_cos_margin,
                        'distmat_margin': args.vf_distmat_margin,
                        'cos_weight': args.vf_cos_weight,
                        'distmat_weight': args.vf_distmat_weight,
                        'vf_max_tokens': args.vf_max_tokens,
                    }
                torch.save(ckpt, os.path.join(args.save_dir, 'last.pt'))
                if current_loss == best_loss:
                    torch.save(ckpt, os.path.join(args.save_dir, 'best.pt'))

    if is_main():
        pbar.close()
        print(f'\nTraining complete: {step} steps, {epoch} epochs')

    if use_ddp:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description='Train CQAR AE with dense patch SR supervision'
    )
    parser.add_argument('--train-dir', required=True, nargs='+')
    parser.add_argument('--save-dir', default='./checkpoints/cqar_ae_sr')
    parser.add_argument('--inp-size', type=int, default=256, help='Fixed LR input size')
    parser.add_argument('--patch-size', type=int, default=256, help='Dense HR supervision patch size')
    parser.add_argument('--scale-min', type=float, default=1.0)
    parser.add_argument('--scale-max', type=float, default=4.0)
    parser.add_argument('--feat-channels', type=int, default=64)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument(
        '--renderer-block-type',
        choices=['window_attention'],
        default='window_attention',
        help='CQAR window-attention renderer.',
    )
    parser.add_argument('--renderer-num-blocks', type=int, default=2)
    parser.add_argument('--renderer-num-heads', type=int, default=4)
    parser.add_argument('--renderer-ffn-expansion-factor', type=float, default=2.0)
    parser.add_argument('--renderer-window-size', type=int, default=8)
    parser.add_argument(
        '--renderer-attention-bias',
        dest='renderer_attention_bias',
        action='store_true',
        help='Enable biases in the window-attention qkv / projection layers.',
    )
    parser.add_argument(
        '--no-renderer-attention-bias',
        dest='renderer_attention_bias',
        action='store_false',
        help='Disable biases in the window-attention path.',
    )
    parser.add_argument(
        '--renderer-layer-norm-bias',
        dest='renderer_layer_norm_bias',
        action='store_true',
        help='Use affine bias in token layer normalization.',
    )
    parser.add_argument(
        '--no-renderer-layer-norm-bias',
        dest='renderer_layer_norm_bias',
        action='store_false',
        help='Disable affine bias in token layer normalization.',
    )
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--max-steps', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--lpips-weight', type=float, default=0.1)
    parser.add_argument('--save-every', type=int, default=5000)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--shuffle-buffer', type=int, default=1000)
    parser.add_argument(
        '--webdataset-probe-records',
        type=int,
        default=64,
        help='Probe this many WebDataset records on the main process before training starts; set 0 to disable.',
    )
    parser.add_argument('--grad-clip-norm', type=float, default=1.0)
    parser.add_argument('--resume', default=None)
    parser.add_argument(
        '--min-size',
        '--resolution',
        dest='min_size',
        type=int,
        default=0,
        help='Optional minimum source-image size filter before random cropping.',
    )
    parser.add_argument('--exts', default='jpg,jpeg,png,webp')
    parser.add_argument('--vf-loss-weight', type=float, default=0.0)
    parser.add_argument('--vf-cos-margin', type=float, default=0.5)
    parser.add_argument('--vf-distmat-margin', type=float, default=0.25)
    parser.add_argument('--vf-cos-weight', type=float, default=1.0)
    parser.add_argument('--vf-distmat-weight', type=float, default=1.0)
    parser.add_argument('--vf-max-tokens', type=int, default=256)
    parser.add_argument(
        '--vf-teacher-type',
        choices=['dino'],
        default='dino',
        help='Vision foundation teacher used by VF loss. Defaults to the previous DINOv2 path.',
    )
    parser.add_argument(
        '--vf-proj-direction',
        choices=['latent_to_teacher', 'teacher_to_latent'],
        default='latent_to_teacher',
    )
    parser.add_argument(
        '--vf-dino-weight-path',
        type=str,
        default=None,
        help='Optional local DINOv2 checkpoint path.',
    )
    parser.add_argument(
        '--vf-dino-repo-dir',
        type=str,
        default=None,
        help='Optional local DINOv2 repo directory.',
    )
    parser.add_argument(
        '--vf-teacher-dtype',
        choices=['float32', 'float16', 'bfloat16'],
        default='float32',
    )
    parser.add_argument(
        '--vf-base-patch-size',
        type=int,
        default=16,
    )
    parser.add_argument(
        '--vf-adaptive-weight',
        dest='vf_adaptive_weight',
        action='store_true',
        help='Use gradient-ratio adaptive weighting for VF loss.',
    )
    parser.add_argument(
        '--no-vf-adaptive-weight',
        dest='vf_adaptive_weight',
        action='store_false',
        help='Disable adaptive weighting and use --vf-loss-weight directly.',
    )
    parser.add_argument('--augment', dest='augment', action='store_true')
    parser.add_argument('--no-augment', dest='augment', action='store_false')
    parser.set_defaults(
        augment=True,
        vf_adaptive_weight=True,
        renderer_attention_bias=False,
        renderer_layer_norm_bias=True,
    )
    args = parser.parse_args()

    if args.scale_max < args.scale_min:
        raise ValueError('--scale-max must be >= --scale-min')
    if args.patch_size > int(round(args.inp_size * args.scale_max)):
        raise ValueError('--patch-size must not exceed the maximum possible HR crop size.')
    if args.renderer_num_blocks < 1:
        raise ValueError('--renderer-num-blocks must be >= 1')
    if args.renderer_num_heads < 1:
        raise ValueError('--renderer-num-heads must be >= 1')
    if args.renderer_ffn_expansion_factor <= 0:
        raise ValueError('--renderer-ffn-expansion-factor must be > 0')
    if args.renderer_window_size < 1:
        raise ValueError('--renderer-window-size must be >= 1')
    if args.hidden % args.renderer_num_heads != 0:
        raise ValueError('--hidden must be divisible by --renderer-num-heads for window_attention blocks')

    train(args)


if __name__ == '__main__':
    main()
