"""Build resize-only gallery assets. Never overwrites the original PNG files."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from html.parser import HTMLParser
import json
from pathlib import Path
import time
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SIZES = list(range(512, 4097, 64))

class GallerySources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'img' and 'data-full' in attrs:
            self.sources.append(attrs['data-full'])

def build(source):
    name = Path(source).stem
    target = ROOT / 'gallery' / 'resolutions' / name
    target.mkdir(parents=True, exist_ok=True)
    frames = []
    with Image.open(ROOT / source) as original:
        original.load()
        original_size = original.size
        if max(original_size) != 4096:
            raise ValueError(f'{source}: expected a 4096px long edge, got {original_size}')
        original = original.convert('RGB')
        for size in SIZES:
            width, height = [max(1, round(d * size / 4096)) for d in original_size]
            full = target / f'{size}.webp'
            preview = target / f'{size}-preview.webp'
            if not full.exists() or not preview.exists():
                resized = original.resize((width, height), Image.Resampling.LANCZOS)
                if not full.exists():
                    resized.save(full, 'WEBP', quality=85, method=4)
                if not preview.exists():
                    resized.thumbnail((768, 768), Image.Resampling.LANCZOS)
                    resized.save(preview, 'WEBP', quality=74, method=4)
            # Metadata and decode checks only; no visual inspection.
            with Image.open(full) as check:
                assert check.size == (width, height), full
                check.verify()
            with Image.open(preview) as check:
                assert max(check.size) <= 768, preview
                check.verify()
            frames.append(dict(size=size, width=width, height=height,
                full=full.relative_to(ROOT).as_posix(),
                preview=preview.relative_to(ROOT).as_posix(),
                fullBytes=full.stat().st_size, previewBytes=preview.stat().st_size))
    return source, dict(original=source, width=original_size[0], height=original_size[1], frames=frames)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    sources = GallerySources()
    sources.feed((ROOT / 'index.html').read_text())
    entries = list(dict.fromkeys(sources.sources))
    assert entries, 'No gallery source images found'
    start = time.monotonic()
    samples = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(build, entry) for entry in entries]
        for job in as_completed(jobs):
            source, sample = job.result()
            samples[source] = sample
            print(f'{len(samples)}/{len(entries)}: {source}, {len(sample["frames"])} sizes', flush=True)
    manifest = dict(version=1, kind='resized-original', min=512, max=4096, step=64,
                    previewMaxEdge=768, samples={key:samples[key] for key in entries})
    path = ROOT / 'gallery' / 'resolutions' / 'manifest.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, separators=(',', ':')) + '\n')
    temporary.replace(path)
    path.with_suffix(".js").write_text("window.RAPD_RESOLUTION_MANIFEST=" + json.dumps(manifest, separators=(",", ":")) + ";\n")
    full_bytes = sum(f['fullBytes'] for s in samples.values() for f in s['frames'])
    preview_bytes = sum(f['previewBytes'] for s in samples.values() for f in s['frames'])
    print(json.dumps(dict(samples=len(samples), frames=len(samples)*len(SIZES),
        fullMiB=round(full_bytes/1024**2,2), previewMiB=round(preview_bytes/1024**2,2),
        elapsedSeconds=round(time.monotonic()-start)), indent=2), flush=True)

if __name__ == '__main__':
    main()
