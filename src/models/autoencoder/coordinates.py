"""Shared coordinate grids for CQAR."""
import torch

def make_coord_grid(shape, value_range=(-1, 1), device='cpu'):
    axes = []
    for i, s in enumerate(shape):
        l = (0.5 + torch.arange(s, device=device)) / s
        if isinstance(value_range[0], (list, tuple)):
            lo, hi = value_range[i]
        else:
            lo, hi = value_range
        l = lo + (hi - lo) * l
        axes.append(l)
    return torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)

def make_coord_cell_grid(shape, value_range=(-1, 1), device='cpu', bs=None):
    coord = make_coord_grid(shape, value_range=value_range, device=device)
    cell = torch.ones_like(coord)
    for i, s in enumerate(shape):
        if isinstance(value_range[0], (list, tuple)):
            lo, hi = value_range[i]
        else:
            lo, hi = value_range
        cell[..., i] *= (hi - lo) / s
    if bs is not None:
        coord = coord.unsqueeze(0).expand(bs, *([-1] * coord.dim()))
        cell = cell.unsqueeze(0).expand(bs, *([-1] * cell.dim()))
    cache_key = (
        tuple(int(s) for s in shape),
        tuple(
            tuple(float(x) for x in item)
            if isinstance(item, (list, tuple))
            else float(item)
            for item in value_range
        ) if isinstance(value_range[0], (list, tuple)) else tuple(float(x) for x in value_range),
    )
    try:
        coord._cqar_grid_cache_key = cache_key
        cell._cqar_grid_cache_key = cache_key
        coord._cqar_grid_batch_mode = 'shared'
        cell._cqar_grid_batch_mode = 'shared'
    except Exception:
        pass
    return coord, cell

