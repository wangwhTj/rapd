"""Mainline CQAR latent flow matching with DINOv2 feature alignment.

The proj.0.* parameter names are retained for checkpoint compatibility.
"""
from typing import Callable

import torch
import torch.nn as nn
from src.utils.no_grad import freeze_model
from src.diffusion.base.training import BaseTrainer
from src.diffusion.base.scheduling import BaseScheduler


def constant(alpha, sigma):
    return 1


def time_shift_fn(t, timeshift=1.0):
    return t / (t + (1 - t) * timeshift)


class RaPDTrainer(BaseTrainer):
    """Train the denoiser and feature projection; keep DINOv2 frozen."""

    def __init__(
        self,
        scheduler: BaseScheduler,
        encoder: nn.Module,
        loss_weight_fn: Callable = constant,
        feat_loss_weight: float = 0.5,
        lognorm_t=False,
        timeshift=1.0,
        align_layer=8,
        proj_denoiser_dim=256,
        proj_hidden_dim=256,
        proj_encoder_dim=256,
        P_mean=-0.8,
        P_std=0.8,
        t_eps=0.05,
        noise_scale: float = 1.0,
        null_condition_p=0.1,
    ):
        super().__init__(null_condition_p=null_condition_p)
        self.lognorm_t = lognorm_t
        self.scheduler = scheduler
        self.timeshift = timeshift
        self.loss_weight_fn = loss_weight_fn
        self.feat_loss_weight = feat_loss_weight
        self.align_layer = align_layer
        self.encoder = encoder
        freeze_model(self.encoder)
        self.proj = nn.Sequential(
            nn.Sequential(
                nn.Linear(proj_denoiser_dim, proj_hidden_dim),
                nn.SiLU(),
                nn.Linear(proj_hidden_dim, proj_hidden_dim),
                nn.SiLU(),
                nn.Linear(proj_hidden_dim, proj_encoder_dim),
            )
        )
        self.P_mean = P_mean
        self.P_std = P_std
        self.t_eps = t_eps
        self.dino_layers = [11]
        self.noise_scale = noise_scale

    def _sample_timestep(self, batch_size, device):
        if self.lognorm_t:
            base_t = (
                torch.randn(batch_size, device=device, dtype=torch.float32)
                * self.P_std + self.P_mean
            ).sigmoid()
        else:
            base_t = torch.rand(batch_size, device=device, dtype=torch.float32)
        return time_shift_fn(base_t, self.timeshift)

    def _impl_trainstep(self, net, ema_net, solver, x, y, metadata=None):
        """Training step for normalized CQAR latent diffusion.

        Args:
            net: JiT_T2I denoiser (DDP-wrapped during training)
            ema_net: EMA copy (not used in loss computation)
            solver: diffusion sampler (not used here)
            x: normalised latent z_norm (B,C,h,w) -- encoded by the frozen AE in main.py
            y: text condition (B,L,D) after null-masking
            metadata: dict with 'raw_image' (B,3,H,W) and 'global_step'
        """
        raw_images = metadata['raw_image']
        batch_size = x.shape[0]

        # 1. Sample timestep
        t = self._sample_timestep(batch_size, x.device)

        # 2. Forward noising: x_t = alpha*x + sigma*noise
        noise = self.noise_scale * torch.randn_like(x)
        alpha = self.scheduler.alpha(t)
        sigma = self.scheduler.sigma(t)
        x_t = alpha * x + noise * sigma

        # 3. Flow matching target velocity
        v_t = (x - x_t) / (1 - t.view(-1, 1, 1, 1)).clamp_min(self.t_eps)

        # 4. Denoiser forward
        if self.feat_loss_weight > 0:
            pred_z0, src_feature = net(x_t, t, y, return_layer=self.align_layer)
            src_feature = self.proj(src_feature)
        else:
            pred_z0 = net(x_t, t, y)
        out = (pred_z0 - x_t) / (1 - t.view(-1, 1, 1, 1)).clamp_min(self.t_eps)

        # 5. REPA cosine alignment loss (skipped when feat_loss_weight == 0)
        if self.feat_loss_weight > 0:
            with torch.no_grad():
                dst_features = self.encoder.get_intermediate_feats(raw_images, n=self.dino_layers)
            if src_feature.shape[1] != dst_features[-1].shape[1]:
                src_feature = torch.nn.functional.interpolate(
                    src_feature.transpose(1, 2),
                    size=dst_features[-1].shape[1],
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            cos_sim = torch.nn.functional.cosine_similarity(
                src_feature, dst_features[-1], dim=-1
            )
            cos_loss_mean = (1 - cos_sim).mean()
        else:
            cos_loss_mean = torch.tensor(0.0, device=x.device)

        # 6. FM loss
        weight = self.loss_weight_fn(alpha, sigma)
        fm_loss = weight * (out - v_t) ** 2
        rec_loss = fm_loss.mean()

        final_loss = rec_loss + self.feat_loss_weight * cos_loss_mean
        return dict(fm_loss=rec_loss, cos_loss=cos_loss_mean, loss=final_loss)
