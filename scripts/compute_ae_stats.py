"""Compute per-channel mean/std of deterministic AE features across a dataset."""

import argparse
import os
import sys

import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

Image.MAX_IMAGE_PIXELS = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.autoencoder.encoder import EDSREncoder


def get_ckpt_value(ckpt_args, override, key, default):
    if override is not None:
        return override
    if isinstance(ckpt_args, dict) and key in ckpt_args:
        return ckpt_args[key]
    return default


# ---------------------------------------------------------------------------
# Dataset (identical to compute_ae_stats.py)
# ---------------------------------------------------------------------------


def resize_short_edge(pil_image, image_size=256):
    """Resize the short edge before center cropping."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    return pil_image


def center_crop_square(img, resolution):
    w, h = img.size
    s = min(w, h)
    return img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))


def preprocess_image(img, resolution, mode='crop_resize'):
    if mode == 'crop_resize':
        img = center_crop_square(img, resolution)
        img = img.resize((resolution, resolution), Image.LANCZOS)
        return img
    if mode == 'resize_crop':
        img = resize_short_edge(img, resolution)
        return center_crop_square(img, resolution)
    raise ValueError(f'Unsupported preprocess mode: {mode}')

class ImageDataset(Dataset):
    def __init__(self, root_dir, resolution=512, max_samples=None,
                 exts='jpg,jpeg,png,webp', preprocess='crop_resize'):
        root = Path(root_dir)
        ext_list = [e.strip().lower().lstrip('.') for e in exts.split(',') if e.strip()]
        files = []
        for ext in ext_list:
            files.extend(root.rglob(f'*.{ext}'))
            files.extend(root.rglob(f'*.{ext.upper()}'))
        files = sorted(set(files))
        if max_samples and len(files) > max_samples:
            import random; random.shuffle(files)
            files = files[:max_samples]
        self.files = files
        if not self.files:
            raise ValueError(f'No images found under: {root_dir}')
        self.resolution = resolution
        self.preprocess = preprocess
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.files[idx]).convert('RGB')
        except Exception:
            img = Image.new('RGB', (self.resolution, self.resolution))
        img = preprocess_image(img, self.resolution, mode=self.preprocess)
        return self.to_tensor(img)


def make_webdataset_loader(dirs, resolution, batch_size, num_workers, num_samples, preprocess_mode):
    import webdataset as wds
    import glob as glob_mod
    tar_paths = []
    for d in dirs:
        tars = sorted(glob_mod.glob(os.path.join(d, '*.tar')))
        tar_paths.extend(tars)
    if not tar_paths:
        raise ValueError(f'No tar files found in: {dirs}')
    print(f'Found {len(tar_paths)} tar files')

    to_tensor_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    def preprocess(img):
        img = preprocess_image(img, resolution, mode=preprocess_mode)
        return to_tensor_norm(img)

    dataset = (
        wds.WebDataset(tar_paths, resampled=False)
        .shuffle(2000)
        .decode('pil', handler=wds.warn_and_continue)
        .to_tuple('jpg;jpeg;png', handler=wds.warn_and_continue)
        .map_tuple(preprocess)
        .batched(batch_size, partial=True)
    )
    loader = wds.WebLoader(dataset, batch_size=None, num_workers=num_workers)
    return loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Compute CQAR encoder feature statistics')
    parser.add_argument('--ckpt', required=True, help='Path to AE checkpoint (last.pt / best.pt)')
    parser.add_argument('--data-dir', required=True, nargs='+', help='Image directory (one or more paths)')
    parser.add_argument('--out', default='./checkpoints/ae_stats.pt', help='Output stats file')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--num-samples', type=int, default=50000, help='Max images to use')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--feat-channels', type=int, default=None)
    parser.add_argument('--ae-type', choices=['auto', 'liif_ae', 'cqar_ae'],
                        default=None, help=argparse.SUPPRESS)
    parser.add_argument('--webdataset', action='store_true', help='Use WebDataset (tar) input instead of folder')
    parser.add_argument(
        '--preprocess',
        choices=['crop_resize', 'resize_crop'],
        default='crop_resize',
        help=(
            'Image preprocessing mode: crop_resize keeps the historical behavior '
            '(center crop square, then resize), while resize_crop matches '
            'short-edge resizing followed by center cropping.'
        ),
    )
    args = parser.parse_args()
    if args.ae_type is not None:
        print("[Deprecated] --ae-type is ignored; statistics use the CQAR deterministic encoder.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print(f'Preprocess mode: {args.preprocess}')

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    ckpt_args = ckpt.get('args', {}) if isinstance(ckpt, dict) else {}
    resolved_feat_channels = get_ckpt_value(ckpt_args, args.feat_channels, 'feat_channels', 64)

    # Load AE encoder weights.
    # Use the same deterministic encoder as CQARAutoencoder.
    encoder = EDSREncoder(out_channels=resolved_feat_channels).to(device).eval()
    encoder_sd = ckpt.get('encoder', ckpt.get('state_dict', ckpt))
    encoder.load_state_dict(encoder_sd, strict=True)
    print(f'Loaded AE encoder from: {args.ckpt}')
    print(f'Resolved feat_channels: {resolved_feat_channels}')

    # Dataset
    if args.webdataset:
        loader = make_webdataset_loader(
            args.data_dir, args.resolution, args.batch_size, args.num_workers,
            args.num_samples, args.preprocess
        )
        total_target = args.num_samples
        print(f'Using WebDataset from: {args.data_dir}')
    else:
        data_dir = args.data_dir[0] if isinstance(args.data_dir, list) else args.data_dir
        ds = ImageDataset(
            data_dir,
            resolution=args.resolution,
            max_samples=args.num_samples,
            preprocess=args.preprocess,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        total_target = len(ds)
        print(f'Dataset size: {total_target} images')

    # Welford online algorithm for numerically stable mean/var
    C = resolved_feat_channels
    z_sum  = torch.zeros(C, dtype=torch.float64)
    z_sum2 = torch.zeros(C, dtype=torch.float64)
    total  = 0

    max_batches = (args.num_samples + args.batch_size - 1) // args.batch_size if args.webdataset else None

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc='Computing z stats', total=max_batches)):
            if max_batches is not None and batch_idx >= max_batches:
                break
            if isinstance(batch, (list, tuple)):
                imgs = batch[0]
                if isinstance(imgs, (list, tuple)):
                    imgs = torch.stack(imgs, dim=0)
            else:
                imgs = batch
            imgs = imgs.to(device)

            z = encoder(imgs)  # (B, C, H, W) deterministic features

            B, C_, h, w = z.shape
            z_flat = z.permute(1, 0, 2, 3).reshape(C_, -1).double().cpu()  # (C, B*h*w)
            z_sum  += z_flat.sum(dim=1)
            z_sum2 += (z_flat ** 2).sum(dim=1)
            total  += B * h * w

    mean = (z_sum / total).float()
    var  = (z_sum2 / total - (z_sum / total) ** 2).float().clamp_min(0)
    std  = var.sqrt()

    stats = {'mean': mean, 'std': std}
    torch.save(stats, args.out)
    print(f'Saved ae_stats.pt -> {args.out}')
    print(f'  mean range: [{mean.min():.4f}, {mean.max():.4f}]')
    print(f'  std  range: [{std.min():.4f},  {std.max():.4f}]')


if __name__ == '__main__':
    main()
