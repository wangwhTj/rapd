import os.path
import numpy
from typing import Any, Callable, Dict
from concurrent.futures import ThreadPoolExecutor


class SaveImagesHook:
    def __init__(self, save_dir="val", save_compressed=False):
        self.save_dir = save_dir
        self.save_compressed = save_compressed

    def save_start(self, target_dir: str, is_global_zero: bool = True):
        self.samples = []
        self.target_dir = target_dir
        self.executor_pool = ThreadPoolExecutor(max_workers=8)
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir, exist_ok=True)
        if is_global_zero:
            print(f"Save images to {self.target_dir}")
        self._saved_num = 0

    def save_image(self, images, metadatas):
        if isinstance(images, (list, tuple)):
            images = [image.permute(1, 2, 0).cpu().numpy() for image in images]
        else:
            images = images.permute(0, 2, 3, 1).cpu().numpy()
        for sample, metadata in zip(images, metadatas):
            save_fn = metadata.pop("save_fn", None)
            self.executor_pool.submit(save_fn, sample, metadata, self.target_dir)

    def process_batch(
        self,
        samples,
        batch: Any,
        is_global_zero: bool,
        all_gather_fn: Callable,
    ) -> None:
        xT, y, metadata = batch
        if isinstance(samples, (list, tuple)):
            if not self.save_compressed or self._saved_num < 10:
                self._saved_num += len(samples)
                self.save_image(samples, metadata)
            return

        b, c, h, w = samples.shape
        if not self.save_compressed or self._saved_num < 10:
            self._saved_num += b
            self.save_image(samples, metadata)

        all_samples = all_gather_fn(samples).view(-1, c, h, w)
        if is_global_zero:
            all_samples = all_samples.permute(0, 2, 3, 1).cpu().numpy()
            self.samples.append(all_samples)

    def save_end(self):
        if self.save_compressed and len(self.samples) > 0:
            samples = numpy.concatenate(self.samples)
            numpy.savez(f'{self.target_dir}/output.npz', arr_0=samples)
        self.executor_pool.shutdown(wait=True)
        self.target_dir = None
        self.executor_pool = None
        self.samples = []

    def state_dict(self) -> Dict[str, Any]:
        return dict()
