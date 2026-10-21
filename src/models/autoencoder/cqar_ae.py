"""CQAR deterministic autoencoder for latent diffusion.

The deterministic EDSR encoder and latent-statistics interface are retained.
This release supports the window-attention CQAR renderer.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.autoencoder.base import BaseAE
from src.models.autoencoder.coordinates import make_coord_cell_grid, make_coord_grid
from src.models.autoencoder.encoder import EDSREncoder


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class TokenLayerNorm(nn.Module):
    """LayerNorm over the last dimension for BHWC / BNC token tensors."""

    def __init__(self, dim: int, bias: bool = True, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition BHWC features into flattened windows."""
    batch, height, width, channels = x.shape
    x = x.view(
        batch,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size * window_size, channels)


def window_reverse(windows: torch.Tensor, window_size: int, height: int, width: int) -> torch.Tensor:
    """Restore BHWC features from flattened windows."""
    batch = windows.shape[0] // ((height // window_size) * (width // window_size))
    x = windows.view(
        batch,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        -1,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(batch, height, width, -1)


def torch_is_compiling() -> bool:
    compiler = getattr(torch, 'compiler', None)
    if compiler is not None and hasattr(compiler, 'is_compiling'):
        return bool(compiler.is_compiling())
    dynamo = getattr(torch, '_dynamo', None)
    if dynamo is not None and hasattr(dynamo, 'is_compiling'):
        return bool(dynamo.is_compiling())
    return False


class WindowSelfAttention(nn.Module):
    """Swin-style local self-attention with relative position bias."""

    def __init__(self, dim: int, num_heads: int, window_size: int, bias: bool = False,
                 use_sdpa: bool = False, sdpa_max_batch_windows: int = 32768):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'Expected dim ({dim}) to be divisible by num_heads ({num_heads}).')
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_sdpa = bool(use_sdpa)
        self.sdpa_max_batch_windows = int(sdpa_max_batch_windows)

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self._relative_position_bias_cache = {}

        table_size = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(table_size, num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(dim=-1)
        self.register_buffer('relative_position_index', relative_position_index, persistent=False)

    def clear_inference_cache(self):
        self._relative_position_bias_cache.clear()

    def _relative_position_bias(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        cache_key = (dtype, device.type, device.index)
        if not torch.is_grad_enabled():
            cached = self._relative_position_bias_cache.get(cache_key)
            if cached is not None:
                return cached

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            self.num_heads,
        ).permute(2, 0, 1).to(dtype=dtype, device=device)
        if not torch.is_grad_enabled():
            relative_position_bias = relative_position_bias.contiguous().detach()
            self._relative_position_bias_cache[cache_key] = relative_position_bias
        return relative_position_bias

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        batch_windows, token_count, channels = x.shape
        qkv = self.qkv(x).view(batch_windows, token_count, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        use_sdpa = (
            self.use_sdpa
            and hasattr(F, 'scaled_dot_product_attention')
            and not torch_is_compiling()
            and (self.sdpa_max_batch_windows <= 0 or batch_windows <= self.sdpa_max_batch_windows)
        )
        if use_sdpa:
            relative_position_bias = self._relative_position_bias(q.dtype, q.device)
            sdpa_mask = relative_position_bias.unsqueeze(0)
            if attn_mask is not None:
                num_windows = attn_mask.shape[0]
                window_mask = attn_mask.to(dtype=q.dtype).unsqueeze(1)
                group_count = batch_windows // num_windows
                sdpa_mask = sdpa_mask + window_mask
                sdpa_mask = sdpa_mask.unsqueeze(0).expand(group_count, -1, -1, -1, -1)
                sdpa_mask = sdpa_mask.reshape(batch_windows, self.num_heads, token_count, token_count)
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=sdpa_mask,
                dropout_p=0.0,
                scale=self.scale,
            )
            out = out.transpose(1, 2).reshape(batch_windows, token_count, channels)
            return self.proj(out)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self._relative_position_bias(attn.dtype, attn.device)
        attn = attn + relative_position_bias.unsqueeze(0)

        if attn_mask is not None:
            num_windows = attn_mask.shape[0]
            attn = attn.view(batch_windows // num_windows, num_windows, self.num_heads, token_count, token_count)
            attn = attn + attn_mask.to(dtype=attn.dtype).unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, token_count, token_count)

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(batch_windows, token_count, channels)
        return self.proj(out)


class GatedTokenFeedForward(nn.Module):
    """Convolution-free gated FFN used by the window-attention renderer."""

    def __init__(self, dim: int, expansion_factor: float = 2.0, bias: bool = False):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.project_in = nn.Linear(dim, hidden_dim * 2, bias=bias)
        self.project_out = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.project_in(x).chunk(2, dim=-1)
        return self.project_out(F.gelu(x1) * x2)


class WindowTransformerBlock(nn.Module):
    """Swin-style local transformer block over dense CQAR query grids."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        window_size: int = 8,
        shift_size: int = 0,
        ffn_expansion_factor: float = 2.0,
        bias: bool = False,
        layer_norm_bias: bool = True,
        use_sdpa_attention: bool = False,
        sdpa_max_batch_windows: int = 32768,
    ):
        super().__init__()
        if window_size < 1:
            raise ValueError(f'Expected window_size >= 1, got {window_size}.')
        if shift_size < 0 or shift_size >= window_size:
            raise ValueError(f'Expected 0 <= shift_size < window_size, got {shift_size} and {window_size}.')
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = TokenLayerNorm(dim, bias=layer_norm_bias)
        self.attn = WindowSelfAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            bias=bias,
            use_sdpa=use_sdpa_attention,
            sdpa_max_batch_windows=sdpa_max_batch_windows,
        )
        self.norm2 = TokenLayerNorm(dim, bias=layer_norm_bias)
        self.ffn = GatedTokenFeedForward(dim=dim, expansion_factor=ffn_expansion_factor, bias=bias)
        self._attn_mask_cache = {}

    def _pad_to_windows(self, x: torch.Tensor):
        _, height, width, _ = x.shape
        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        return x, pad_h, pad_w

    def clear_inference_cache(self):
        self._attn_mask_cache.clear()
        self.attn.clear_inference_cache()

    def _make_attn_mask(self, height: int, width: int, device: torch.device,
                        dtype: torch.dtype) -> torch.Tensor:
        if self.shift_size == 0:
            return None
        cache_key = (height, width, dtype, device.type, device.index)
        if not torch.is_grad_enabled():
            cached = self._attn_mask_cache.get(cache_key)
            if cached is not None:
                return cached

        img_mask = torch.zeros((1, height, width, 1), device=device, dtype=dtype)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        count = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = count
                count += 1
        mask_windows = window_partition(img_mask, self.window_size).squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float('-100.0')).masked_fill(attn_mask == 0, 0.0)
        if not torch.is_grad_enabled():
            attn_mask = attn_mask.contiguous().detach()
            self._attn_mask_cache[cache_key] = attn_mask
        return attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x, pad_h, pad_w = self._pad_to_windows(x)
        _, padded_h, padded_w, _ = x.shape

        shift_size = 0 if min(shortcut.shape[1], shortcut.shape[2]) <= self.window_size else self.shift_size
        attn_mask = self._make_attn_mask(padded_h, padded_w, x.device, x.dtype) if shift_size > 0 else None

        if shift_size > 0:
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))

        x_windows = window_partition(x, self.window_size)
        attn_windows = self.attn(x_windows, attn_mask=attn_mask)
        x = window_reverse(attn_windows, self.window_size, padded_h, padded_w)

        if shift_size > 0:
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))
        if pad_h > 0 or pad_w > 0:
            x = x[:, :shortcut.shape[1], :shortcut.shape[2], :].contiguous()

        x = shortcut + x
        x = x + self.ffn(self.norm2(x))
        return x


def query_local_features(feat: torch.Tensor, coord: torch.Tensor, cell: torch.Tensor):
    """Official INFD geometry helper for CQAR queries."""
    batch = feat.shape[0]
    device = feat.device
    query_shape = coord.shape[1:-1]

    coord = coord.view(batch, 1, -1, 2)
    cell = cell.view(batch, 1, -1, 2)

    feat_coord = make_coord_grid(feat.shape[-2:], device=device)
    feat_coord = feat_coord.permute(2, 0, 1).unsqueeze(0).expand(batch, -1, -1, -1)

    q_feat = F.grid_sample(
        feat, coord.flip(-1), mode='nearest', align_corners=False
    ).permute(0, 2, 3, 1)
    q_coord = F.grid_sample(
        feat_coord, coord.flip(-1), mode='nearest', align_corners=False
    ).permute(0, 2, 3, 1)

    rel_coord = coord - q_coord
    rel_coord[..., 0] *= feat.shape[-2]
    rel_coord[..., 1] *= feat.shape[-1]

    rel_cell = cell.clone()
    rel_cell[..., 0] *= feat.shape[-2]
    rel_cell[..., 1] *= feat.shape[-1]

    q_feat = q_feat.view(batch, *query_shape, -1)
    rel_coord = rel_coord.view(batch, *query_shape, 2)
    rel_cell = rel_cell.view(batch, *query_shape, 2)
    return q_feat, rel_coord, rel_cell


class WindowAttentionRendererNet(nn.Module):
    """Convolution-free CQAR tail built from local window-attention blocks."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 3,
        num_blocks: int = 2,
        num_heads: int = 4,
        ffn_expansion_factor: float = 2.0,
        attention_bias: bool = False,
        layer_norm_bias: bool = True,
        window_size: int = 8,
        use_sdpa_attention: bool = False,
        sdpa_max_batch_windows: int = 32768,
    ):
        super().__init__()
        if num_blocks < 1:
            raise ValueError(f'Expected num_blocks >= 1, got {num_blocks}.')
        self.input_proj = nn.Linear(in_channels, hidden_channels, bias=attention_bias)
        self.blocks = nn.ModuleList()
        for index in range(num_blocks):
            shift_size = window_size // 2 if window_size > 1 and index % 2 == 1 else 0
            self.blocks.append(
                WindowTransformerBlock(
                    dim=hidden_channels,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=attention_bias,
                    layer_norm_bias=layer_norm_bias,
                    use_sdpa_attention=use_sdpa_attention,
                    sdpa_max_batch_windows=sdpa_max_batch_windows,
                )
        )
        self.norm_out = TokenLayerNorm(hidden_channels, bias=layer_norm_bias)
        self.output_proj = nn.Linear(hidden_channels, out_channels, bias=True)
        self.active_num_blocks = len(self.blocks)

    def set_active_num_blocks(self, count: int):
        self.active_num_blocks = max(0, min(int(count), len(self.blocks)))

    def get_active_num_blocks(self):
        return self.active_num_blocks, len(self.blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for index, block in enumerate(self.blocks):
            if index >= self.active_num_blocks:
                break
            x = block(x)
        x = nonlinearity(self.norm_out(x))
        x = self.output_proj(x)
        return x.permute(0, 3, 1, 2).contiguous()

    def get_last_layer_weight(self) -> torch.Tensor:
        return self.output_proj.weight

    def clear_inference_cache(self):
        for block in self.blocks:
            if hasattr(block, 'clear_inference_cache'):
                block.clear_inference_cache()


class CQARRenderer(nn.Module):
    """Coordinate-queried renderer with configurable dense-grid rendering blocks."""

    def __init__(
        self,
        feat_channels: int = 64,
        hidden_channels: int = 256,
        block_type: str = 'window_attention',
        num_blocks: int = 2,
        num_heads: int = 4,
        ffn_expansion_factor: float = 2.0,
        attention_bias: bool = False,
        layer_norm_bias: bool = True,
        window_size: int = 8,
        geometry_cache_size: int = 16,
        use_sdpa_attention: bool = False,
        sdpa_max_batch_windows: int = 32768,
    ):
        super().__init__()
        self.block_type = block_type
        self.use_absolute_coord = True
        self.geometry_cache_size = int(geometry_cache_size)
        self._geometry_cache = {}
        in_channels = feat_channels + 6
        if block_type != 'window_attention':
            raise ValueError('Only window_attention is supported in this release.')
        self.net = WindowAttentionRendererNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=3,
            num_blocks=num_blocks,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            attention_bias=attention_bias,
            layer_norm_bias=layer_norm_bias,
            window_size=window_size,
            use_sdpa_attention=use_sdpa_attention,
            sdpa_max_batch_windows=sdpa_max_batch_windows,
        )

    def _reshape_query_grid(self, coord: torch.Tensor, cell: torch.Tensor):
        if coord.dim() == 4:
            return coord, cell, False
        if coord.dim() != 3:
            raise ValueError(f'Expected coord with 3 or 4 dims, got shape {tuple(coord.shape)}.')

        batch, query_count, _ = coord.shape
        side = int(math.isqrt(query_count))
        if side * side != query_count:
            raise ValueError(
                'CQARRenderer requires dense grid queries. '
                f'Got flattened query count {query_count}, which is not a square grid.'
            )
        coord = coord.view(batch, side, side, 2)
        cell = cell.view(batch, side, side, 2)
        return coord, cell, True

    def get_last_layer_weight(self) -> torch.Tensor:
        return self.net.get_last_layer_weight()

    def set_active_num_blocks(self, count: int):
        if not hasattr(self.net, 'set_active_num_blocks'):
            raise AttributeError(f'{type(self.net).__name__} does not support active block selection.')
        self.net.set_active_num_blocks(count)

    def get_active_num_blocks(self):
        if hasattr(self.net, 'get_active_num_blocks'):
            return self.net.get_active_num_blocks()
        return None, None

    def clear_geometry_cache(self):
        self._geometry_cache.clear()

    def clear_inference_cache(self):
        self.clear_geometry_cache()
        if hasattr(self.net, 'clear_inference_cache'):
            self.net.clear_inference_cache()

    @staticmethod
    def _copy_grid_cache_attrs(dst: torch.Tensor, src: torch.Tensor):
        for attr in ('_cqar_grid_cache_key', '_cqar_grid_batch_mode'):
            if hasattr(src, attr):
                try:
                    setattr(dst, attr, getattr(src, attr))
                except Exception:
                    pass
        return dst

    def _get_geometry_cache_key(self, feat: torch.Tensor, coord: torch.Tensor, cell: torch.Tensor):
        if self.geometry_cache_size <= 0 or torch.is_grad_enabled():
            return None, None
        coord_key = getattr(coord, '_cqar_grid_cache_key', None)
        cell_key = getattr(cell, '_cqar_grid_cache_key', None)
        if coord_key is None or coord_key != cell_key:
            return None, None
        batch_mode = getattr(coord, '_cqar_grid_batch_mode', 'shared')
        if batch_mode != getattr(cell, '_cqar_grid_batch_mode', batch_mode):
            return None, None
        if batch_mode not in {'shared', 'batched'}:
            return None, None
        geom_batch = 1 if batch_mode == 'shared' else coord.shape[0]
        key = (
            coord_key,
            batch_mode,
            geom_batch,
            tuple(int(x) for x in feat.shape[-2:]),
            tuple(int(x) for x in coord.shape[1:-1]),
            feat.device.type,
            feat.device.index,
            str(coord.dtype),
        )
        return key, batch_mode

    def _remember_geometry(self, key, geometry):
        if len(self._geometry_cache) >= self.geometry_cache_size:
            self._geometry_cache.pop(next(iter(self._geometry_cache)))
        self._geometry_cache[key] = geometry
        return geometry

    def _build_geometry(self, feat: torch.Tensor, coord: torch.Tensor,
                        cell: torch.Tensor, batch_mode: str):
        if batch_mode == 'shared':
            coord = coord[:1]
            cell = cell[:1]
        batch = coord.shape[0]
        query_shape = coord.shape[1:-1]
        coord_flat = coord.reshape(batch, 1, -1, 2)
        cell_flat = cell.reshape(batch, 1, -1, 2)

        feat_coord = make_coord_grid(feat.shape[-2:], device=feat.device)
        feat_coord = feat_coord.permute(2, 0, 1).unsqueeze(0).expand(batch, -1, -1, -1)
        q_coord = F.grid_sample(
            feat_coord,
            coord_flat.flip(-1),
            mode='nearest',
            align_corners=False,
        ).permute(0, 2, 3, 1)

        rel_coord = coord_flat - q_coord
        rel_coord[..., 0] *= feat.shape[-2]
        rel_coord[..., 1] *= feat.shape[-1]

        rel_cell = cell_flat.clone()
        rel_cell[..., 0] *= feat.shape[-2]
        rel_cell[..., 1] *= feat.shape[-1]

        coord_layout = coord.reshape(batch, *query_shape, 2).contiguous().detach()
        coord_sample = coord_flat.flip(-1).contiguous().detach()
        rel_coord = rel_coord.reshape(batch, *query_shape, 2).contiguous().detach()
        rel_cell = rel_cell.reshape(batch, *query_shape, 2).contiguous().detach()
        return coord_layout, coord_sample, rel_coord, rel_cell

    def _expand_geometry(self, geometry, batch: int):
        coord_layout, coord_sample, rel_coord, rel_cell = geometry
        geom_batch = coord_layout.shape[0]
        if geom_batch == batch:
            return geometry
        if geom_batch != 1:
            return None
        return (
            coord_layout.expand(batch, *coord_layout.shape[1:]),
            coord_sample.expand(batch, *coord_sample.shape[1:]),
            rel_coord.expand(batch, *rel_coord.shape[1:]),
            rel_cell.expand(batch, *rel_cell.shape[1:]),
        )

    def _convert_feat_coord_cell(self, feat: torch.Tensor, coord: torch.Tensor, cell: torch.Tensor):
        cache_key, batch_mode = self._get_geometry_cache_key(feat, coord, cell)
        if cache_key is None:
            q_feat, rel_coord, rel_cell = query_local_features(feat, coord, cell)
            return q_feat, rel_coord, rel_cell, coord

        geometry = self._geometry_cache.get(cache_key)
        if geometry is None:
            geometry = self._remember_geometry(
                cache_key,
                self._build_geometry(feat, coord, cell, batch_mode),
            )
        geometry = self._expand_geometry(geometry, feat.shape[0])
        if geometry is None:
            q_feat, rel_coord, rel_cell = query_local_features(feat, coord, cell)
            return q_feat, rel_coord, rel_cell, coord

        coord_layout, coord_sample, rel_coord, rel_cell = geometry
        q_feat = F.grid_sample(
            feat,
            coord_sample,
            mode='nearest',
            align_corners=False,
        ).permute(0, 2, 3, 1)
        q_feat = q_feat.reshape(feat.shape[0], *coord_layout.shape[1:-1], -1)
        return q_feat, rel_coord, rel_cell, coord_layout

    def forward(self, feat: torch.Tensor, coord: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        input_coord = coord
        input_cell = cell
        coord, cell, flatten_output = self._reshape_query_grid(coord, cell)
        self._copy_grid_cache_attrs(coord, input_coord)
        self._copy_grid_cache_attrs(cell, input_cell)
        q_feat, rel_coord, rel_cell, coord = self._convert_feat_coord_cell(feat, coord, cell)
        layout = [q_feat, rel_coord, rel_cell]
        layout.append(coord)
        rgb = self.net(torch.cat(layout, dim=-1))
        if flatten_output:
            return rgb.flatten(2).transpose(1, 2).contiguous()
        return rgb


class CQARAutoencoder(BaseAE):
    """Deterministic EDSR encoder paired with a CQAR renderer."""

    def __init__(
        self,
        feat_channels: int = 64,
        hidden: int = 256,
        stats_path: str = None,
        use_ae_stats: bool = True,
        renderer_block_type: str = 'window_attention',
        renderer_num_blocks: int = 2,
        renderer_num_heads: int = 4,
        renderer_ffn_expansion_factor: float = 2.0,
        renderer_attention_bias: bool = False,
        renderer_layer_norm_bias: bool = True,
        renderer_window_size: int = 8,
        renderer_geometry_cache_size: int = 16,
        renderer_use_sdpa_attention: bool = False,
        renderer_sdpa_max_batch_windows: int = 32768,
    ):
        super().__init__()
        self.encoder = EDSREncoder(out_channels=feat_channels)
        self.renderer = CQARRenderer(
            feat_channels=feat_channels,
            hidden_channels=hidden,
            block_type=renderer_block_type,
            num_blocks=renderer_num_blocks,
            num_heads=renderer_num_heads,
            ffn_expansion_factor=renderer_ffn_expansion_factor,
            attention_bias=renderer_attention_bias,
            layer_norm_bias=renderer_layer_norm_bias,
            window_size=renderer_window_size,
            geometry_cache_size=renderer_geometry_cache_size,
            use_sdpa_attention=renderer_use_sdpa_attention,
            sdpa_max_batch_windows=renderer_sdpa_max_batch_windows,
        )
        self.use_ae_stats = use_ae_stats

        if use_ae_stats and stats_path is not None:
            stats = torch.load(stats_path, map_location='cpu', weights_only=False)
            z_mean = stats['mean']
            z_std = stats['std']
            if z_mean.dim() == 1:
                z_mean = z_mean.view(-1, 1, 1)
                z_std = z_std.view(-1, 1, 1)
        else:
            z_mean = torch.zeros(feat_channels, 1, 1)
            z_std = torch.ones(feat_channels, 1, 1)

        self.register_buffer('z_mean', z_mean)
        self.register_buffer('z_std', z_std)

    def forward(
        self,
        x: torch.Tensor,
        coord: torch.Tensor = None,
        cell: torch.Tensor = None,
        return_latent: bool = False,
    ):
        z = self.encoder(x)
        if coord is None or cell is None:
            height, width = x.shape[-2:]
            coord, cell = make_coord_cell_grid((height, width), device=x.device, bs=x.shape[0])
        pred = self.renderer(z, coord, cell)
        if return_latent:
            return pred, z
        return pred

    def _impl_encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        if self.use_ae_stats:
            return (z - self.z_mean) / self.z_std.clamp_min(1e-6)
        return z

    def _impl_decode(self, x: torch.Tensor, out_size=None) -> torch.Tensor:
        if self.use_ae_stats:
            z = x * self.z_std + self.z_mean
        else:
            z = x
        if out_size is None:
            out_size = (z.shape[-2], z.shape[-1])
        coord, cell = make_coord_cell_grid(out_size, device=z.device, bs=z.shape[0])
        return self.renderer(z, coord, cell)

    @classmethod
    def load_from_ae_checkpoint(cls, ckpt_path: str, feat_channels: int = 64,
                                hidden: int = 256, stats_path: str = None,
                                use_ae_stats: bool = True,
                                renderer_block_type: str = 'window_attention',
                                renderer_num_blocks: int = 2,
                                renderer_num_heads: int = 4,
                                renderer_ffn_expansion_factor: float = 2.0,
                                renderer_attention_bias: bool = False,
                                renderer_layer_norm_bias: bool = True,
                                renderer_window_size: int = 8,
                                renderer_geometry_cache_size: int = 16,
                                renderer_use_sdpa_attention: bool = False,
                                renderer_sdpa_max_batch_windows: int = 32768) -> 'CQARAutoencoder':
        model = cls(
            feat_channels=feat_channels,
            hidden=hidden,
            stats_path=stats_path,
            use_ae_stats=use_ae_stats,
            renderer_block_type=renderer_block_type,
            renderer_num_blocks=renderer_num_blocks,
            renderer_num_heads=renderer_num_heads,
            renderer_ffn_expansion_factor=renderer_ffn_expansion_factor,
            renderer_attention_bias=renderer_attention_bias,
            renderer_layer_norm_bias=renderer_layer_norm_bias,
            renderer_window_size=renderer_window_size,
            renderer_geometry_cache_size=renderer_geometry_cache_size,
            renderer_use_sdpa_attention=renderer_use_sdpa_attention,
            renderer_sdpa_max_batch_windows=renderer_sdpa_max_batch_windows,
        )
        raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if isinstance(raw, dict) and 'encoder' in raw and 'renderer' in raw:
            model.encoder.load_state_dict(raw['encoder'], strict=True)
            model.renderer.load_state_dict(raw['renderer'], strict=True)
        else:
            enc_sd = {k[len('encoder.'):]: v for k, v in raw.items() if k.startswith('encoder.')}
            ren_sd = {k[len('renderer.'):]: v for k, v in raw.items() if k.startswith('renderer.')}
            if enc_sd:
                model.encoder.load_state_dict(enc_sd, strict=True)
            if ren_sd:
                model.renderer.load_state_dict(ren_sd, strict=True)
        print(f'[CQARAutoencoder] Loaded pretrained weights from: {ckpt_path}')
        return model


