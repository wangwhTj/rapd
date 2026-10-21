import importlib
import random

from torch.utils.data import Dataset


def _instantiate_from_config(cfg):
    class_path = cfg.get("class_path")
    if not class_path:
        raise ValueError("Missing class_path in dataset config")
    module_name, class_name = class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(**cfg.get("init_args", {}))


def _parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


class MixedImageTextLongCaption(Dataset):
    """Map-style dataset that probabilistically replaces samples with long captions."""

    def __init__(
        self,
        main_dataset,
        long_caption=None,
    ):
        super().__init__()
        self.main_dataset = _instantiate_from_config(main_dataset)

        long_caption = long_caption or {}
        self.long_caption_enabled = _parse_bool(long_caption.get("enabled"), default=False)
        self.long_caption_ratio = float(long_caption.get("ratio", 0.0))
        if not 0.0 <= self.long_caption_ratio <= 1.0:
            raise ValueError(f"long_caption.ratio must be in [0, 1], got {self.long_caption_ratio}")

        self.long_caption_dataset = None
        self._long_caption_iter = None
        if self.long_caption_enabled and self.long_caption_ratio > 0.0:
            dataset_cfg = long_caption.get("dataset")
            if not dataset_cfg:
                raise ValueError("long_caption.dataset is required when long_caption.enabled=true")
            self.long_caption_dataset = _instantiate_from_config(dataset_cfg)

    def __len__(self):
        return len(self.main_dataset)

    def _next_long_caption_sample(self):
        if self._long_caption_iter is None:
            self._long_caption_iter = iter(self.long_caption_dataset)

        while True:
            try:
                return next(self._long_caption_iter)
            except StopIteration:
                self._long_caption_iter = iter(self.long_caption_dataset)

    def __getitem__(self, idx):
        if (
            self.long_caption_dataset is not None
            and random.random() < self.long_caption_ratio
        ):
            return self._next_long_caption_sample()
        return self.main_dataset[idx]
