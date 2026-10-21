"""Deterministic encoder used by CQAR; parameter names retain checkpoint compatibility."""
import torch
import torch.nn as nn
class _EDSRResBlock(nn.Module):
    """EDSR-style residual block: Conv -> ReLU -> Conv, with res_scale."""

    def __init__(self, n_feats: int, res_scale: float = 1.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class EDSREncoder(nn.Module):
    """EDSR encoder with a single deterministic 1x1 output projection."""

    def __init__(self, out_channels: int = 64, n_resblocks: int = 8,
                 res_scale: float = 1.0):
        super().__init__()
        C = out_channels
        # EDSR backbone (identical to LIIFEncoder)
        self.head = nn.Conv2d(3, C, 3, padding=1)
        self.body = nn.Sequential(
            *[_EDSRResBlock(C, res_scale) for _ in range(n_resblocks)]
        )
        self.tail = nn.Conv2d(C, C, 3, padding=1)
        # Output projection; retain the mean_head parameter name for existing weights.
        self.mean_head = nn.Conv2d(C, C, 1)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Accept legacy encoder weights without instantiating the removed branch.
        for suffix in ('weight', 'bias'):
            state_dict.pop(prefix + 'logvar_head.' + suffix, None)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.head(x)
        h = self.body(h)
        features = h + self.tail(h)
        mean = self.mean_head(features)
        return mean
