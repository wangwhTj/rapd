"""Shared preprocessing and DINOv2 feature guidance for CQAR AE training."""
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import InterpolationMode, Normalize
from torchvision.transforms import functional as TF


def resize_tensor(img: torch.Tensor, size: int) -> torch.Tensor:
    """Match LIIF's bicubic PIL resize path for tensor images."""
    pil = TF.to_pil_image(img)
    pil = TF.resize(pil, [size, size], interpolation=InterpolationMode.BICUBIC)
    return TF.to_tensor(pil)

def augment_pair(lr: torch.Tensor, hr: torch.Tensor):
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    dflip = random.random() < 0.5

    def apply(x: torch.Tensor) -> torch.Tensor:
        if hflip:
            x = x.flip(-2)
        if vflip:
            x = x.flip(-1)
        if dflip:
            x = x.transpose(-2, -1)
        return x

    return apply(lr), apply(hr)

class FrozenDINOv2Teacher(nn.Module):
    """Frozen DINOv2 ViT-B/14 teacher that returns patch features as a 2D map."""

    def __init__(
        self,
        weight_path: str = None,
        repo_dir: str = None,
        base_patch_size: int = 16,
    ):
        super().__init__()
        expanded_weight_path = None
        if weight_path:
            expanded_weight_path = os.path.expanduser(weight_path)
            if not os.path.isfile(expanded_weight_path):
                raise ValueError(f'Invalid DINOv2 weight path: {weight_path}')

        local_repo = None
        if repo_dir is not None:
            expanded_repo_dir = os.path.expanduser(repo_dir)
            if not os.path.isdir(expanded_repo_dir):
                raise ValueError(f'Invalid DINOv2 repo dir: {repo_dir}')
            local_repo = expanded_repo_dir
        else:
            hub_dir = torch.hub.get_dir()
            if os.path.isdir(hub_dir):
                for name in os.listdir(hub_dir):
                    if name.startswith('facebookresearch_dinov2'):
                        local_repo = os.path.join(hub_dir, name)
                        break

        load_kwargs = {'trust_repo': True}
        if expanded_weight_path is not None:
            load_kwargs['pretrained'] = False

        if local_repo is not None and os.path.isdir(local_repo):
            self.encoder = torch.hub.load(
                local_repo,
                'dinov2_vitb14',
                source='local',
                **load_kwargs,
            )
        else:
            self.encoder = torch.hub.load(
                'facebookresearch/dinov2',
                'dinov2_vitb14',
                **load_kwargs,
            )

        if expanded_weight_path is not None:
            state_dict = torch.load(expanded_weight_path, map_location='cpu', weights_only=False)
            self.encoder.load_state_dict(state_dict, strict=False)

        patch_size = self.encoder.patch_embed.patch_size
        self.patch_size = patch_size[0] if isinstance(patch_size, tuple) else patch_size
        self.base_patch_size = base_patch_size
        self.feature_dim = getattr(self.encoder, 'embed_dim', 768)
        self.normalizer = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad_(False)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.encoder.parameters()).dtype

    def _prepare_pixels(self, x: torch.Tensor) -> torch.Tensor:
        x = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
        height, width = x.shape[-2:]
        target_h = max(self.patch_size, int(self.patch_size * height / self.base_patch_size))
        target_w = max(self.patch_size, int(self.patch_size * width / self.base_patch_size))
        target_h = max(self.patch_size, (target_h // self.patch_size) * self.patch_size)
        target_w = max(self.patch_size, (target_w // self.patch_size) * self.patch_size)
        x = F.interpolate(
            x,
            size=(target_h, target_w),
            mode='bicubic',
            align_corners=False,
            antialias=True,
        )
        x = self.normalizer(x)
        return x.to(dtype=self.dtype)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pixel_values = self._prepare_pixels(x)
        features = self.encoder.forward_features(pixel_values)
        if isinstance(features, dict):
            tokens = features['x_norm_patchtokens']
        else:
            tokens = features[:, 1:]

        grid_h = pixel_values.shape[-2] // self.patch_size
        grid_w = pixel_values.shape[-1] // self.patch_size
        tokens = tokens.reshape(tokens.shape[0], grid_h, grid_w, tokens.shape[-1])
        return tokens.permute(0, 3, 1, 2).contiguous()

def maybe_cast_teacher(teacher, device, teacher_dtype):
    teacher = teacher.to(device)
    if teacher_dtype == 'bfloat16':
        teacher = teacher.to(dtype=torch.bfloat16)
    elif teacher_dtype == 'float16':
        teacher = teacher.to(dtype=torch.float16)
    elif teacher_dtype != 'float32':
        raise ValueError(f'Unsupported teacher dtype: {teacher_dtype}')
    return teacher

def align_latent_to_teacher_grid(latent: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
    teacher_hw = teacher_feat.shape[-2:]
    if latent.shape[-2:] == teacher_hw:
        return latent
    return F.adaptive_avg_pool2d(latent, output_size=teacher_hw)

def maybe_subsample_tokens(z_flat, teacher_flat, max_tokens):
    num_tokens = z_flat.shape[-1]
    if max_tokens is None or max_tokens <= 0 or num_tokens <= max_tokens:
        return z_flat, teacher_flat
    idx = torch.randperm(num_tokens, device=z_flat.device)[:max_tokens]
    return z_flat.index_select(-1, idx), teacher_flat.index_select(-1, idx)

def get_vf_feature_pair(latent, teacher_feat, vf_proj, proj_direction):
    latent = align_latent_to_teacher_grid(latent, teacher_feat)
    if proj_direction == 'latent_to_teacher':
        latent_cmp = vf_proj(latent)
        teacher_cmp = teacher_feat
    elif proj_direction == 'teacher_to_latent':
        latent_cmp = latent
        teacher_cmp = vf_proj(teacher_feat)
    else:
        raise ValueError(f'Unsupported proj_direction: {proj_direction}')
    return latent_cmp.float(), teacher_cmp.float()

def compute_vf_losses(
    latent,
    teacher_feat,
    vf_proj,
    proj_direction='latent_to_teacher',
    cos_margin=0.5,
    distmat_margin=0.25,
    cos_weight=1.0,
    distmat_weight=1.0,
    max_tokens=256,
):
    latent_cmp, teacher_cmp = get_vf_feature_pair(latent, teacher_feat, vf_proj, proj_direction)

    cosine = F.cosine_similarity(latent_cmp, teacher_cmp, dim=1)
    mcos_loss = F.relu(1.0 - cos_margin - cosine).mean()

    z_flat = latent_cmp.flatten(2)
    teacher_flat = teacher_cmp.flatten(2)
    z_flat, teacher_flat = maybe_subsample_tokens(z_flat, teacher_flat, max_tokens)

    z_norm = F.normalize(z_flat, dim=1)
    teacher_norm = F.normalize(teacher_flat, dim=1)
    z_cos = torch.einsum('bcn,bcm->bnm', z_norm, z_norm)
    teacher_cos = torch.einsum('bcn,bcm->bnm', teacher_norm, teacher_norm)
    mdms_loss = F.relu(torch.abs(z_cos - teacher_cos) - distmat_margin).mean()

    vf_loss = cos_weight * mcos_loss + distmat_weight * mdms_loss
    mean_cosine = cosine.mean()
    return mcos_loss, mdms_loss, vf_loss, mean_cosine

def calculate_adaptive_vf_weight(rec_loss, vf_loss, last_layer, base_weight):
    rec_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
    vf_grads = torch.autograd.grad(vf_loss, last_layer, retain_graph=True)[0]
    adaptive = torch.norm(rec_grads) / (torch.norm(vf_grads) + 1e-4)
    adaptive = torch.clamp(adaptive, 0.0, 1e8).detach()
    return adaptive * base_weight
