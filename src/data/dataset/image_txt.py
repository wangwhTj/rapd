import torch
import os

from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Normalize, Resize
from torchvision.transforms.functional import to_tensor
from PIL import Image

EXTs = ['.png', '.jpg', '.jpeg', ".JPEG"]


def is_image_file(filename):
    return any(filename.endswith(ext) for ext in EXTs)

class ImageText(Dataset):
    def __init__(self, root, resolution):
        super().__init__()
        self.root = os.path.abspath(root)
        self.image_paths = []
        self.texts = []
        n_images_seen = 0
        n_missing_txt = 0
        if not os.path.isdir(self.root):
            import sys
            print(f"[ImageText] ERROR: root is not a directory or does not exist: {self.root!r}", file=sys.stderr)
        for dir, subdirs, files in os.walk(self.root):
            for file in files:
                if is_image_file(file):
                    image_path = os.path.join(dir, file)
                    image_base_path = image_path.split(".")[:-1]
                    text_path = ".".join(image_base_path) + ".txt"
                    if os.path.exists(text_path):
                        with open(text_path, 'r') as f:
                            text = f.read()
                        self.texts.append(text)
                        self.image_paths.append(image_path)
                    else:
                        n_missing_txt += 1
                    n_images_seen += 1

        if len(self.image_paths) == 0:
            import sys
            msg = (
                f"[ImageText] root={self.root!r}\n"
                f"  -> 0 samples loaded. "
                f"Images found: {n_images_seen}, skipped (no same-name .txt): {n_missing_txt}.\n"
                f"  -> For each image there must be a .txt with same base name (e.g. a.jpg + a.txt)."
            )
            print(msg, file=sys.stderr)
        elif n_missing_txt > 0:
            import sys
            print(
                f"[ImageText] root={self.root!r} len={len(self.image_paths)} "
                f"(skipped {n_missing_txt} images without .txt)",
                file=sys.stderr,
            )

        self.resize = Resize(resolution)
        self.center_crop = CenterCrop(resolution)
        self.normalize = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        text = self.texts[idx]
        pil_image = Image.open(image_path).convert('RGB')
        pil_image = self.resize(pil_image)
        pil_image = self.center_crop(pil_image)
        raw_image = to_tensor(pil_image)
        normalized_image = self.normalize(raw_image)
        metadata = {
            "image_path": image_path,
            "prompt": text,
            "raw_image": raw_image,
        }
        return normalized_image, text, metadata

    def __len__(self):
        return len(self.image_paths)


