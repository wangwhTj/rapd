"""WebDataset input for the public pretraining and SFT recipes."""
from typing import Dict, Iterable, Optional
import random
import os
import glob
import io
import torch
import webdataset as wds
from PIL import Image
from torch.utils.data import IterableDataset
from torchvision.transforms import Normalize
from torchvision.transforms.functional import to_tensor

def resize(pil_image, image_size=256):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    return pil_image

def center_crop_fn(image, height, width):
    crop_x = (image.width - width) // 2
    crop_y = (image.height - height) // 2
    return image.crop((crop_x, crop_y, crop_x + width, crop_y + height))

def random_crop_fn(image, height, width):
    crop_x = random.randint(0, image.width - width)
    crop_y = random.randint(0, image.height - height)
    return image.crop((crop_x, crop_y, crop_x + width, crop_y + height))

class WebDatasetPackedDataset(IterableDataset):
    """
    A highly efficient WebDataset loader for large-scale pre-training.
    It streams data from .tar files and is optimized for multi-worker data loading.

    This dataset yields tuples of: (normalized_image_tensor, caption_str, metadata_dict)

    Args:
        urls (Iterable[str]): A list of URLs or glob patterns pointing to .tar files.
        resolution (int): The target short-side resolution for image resizing.
        random_crop (bool): If True, applies random cropping; otherwise, center cropping.
        shuffle_buffer (int): The size of the buffer for shuffling samples. A larger buffer
                              provides better randomness but uses more memory. 0 to disable.
        sample_shuffle (bool): If True, enables sample shuffling within the buffer.
        repeat (bool): If True, the dataset will loop indefinitely. Set to True for training.
    """
    def __init__(
        self,
        urls: Iterable[str],
        resolution: int = 256,
        random_crop: bool = False,
        shuffle_buffer: int = 1000,
        sample_shuffle: bool = True,
        repeat: bool = True,
    ):
        super().__init__()
        # Ensure urls is a list, even if a single string is passed
        if isinstance(urls, str):
            urls = [urls]
        print("INFO: Resolving dataset URLs and glob patterns...")
        
        tar_files = []
        for url in urls:
            tar_files.extend(glob.glob(os.path.join(url, "**/*.tar"), recursive=True))
            tar_files.extend(glob.glob(os.path.join(url, "**/*.tar.gz"), recursive=True))
        num_tars = len(tar_files)   
        if num_tars == 0:
            raise ValueError(f"No tar files found. Please check your URLs/patterns: {urls}")
            
        print(f"INFO: Found {num_tars} tar files to stream from.")
        
        self.urls = tar_files

        self.resolution = resolution
        self.normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.crop_fn = random_crop_fn if random_crop else center_crop_fn
        self.shuffle_buffer = shuffle_buffer
        self.sample_shuffle = sample_shuffle
        self.repeat = repeat

        # A simple text augmentation to add variety to short captions
        self.prefix_template = [
            "A photo of ", "A picture of ", "A visual representation of ",
            "A image of ", "A scene of ", "A view of ", "A depiction of ",
        ]

    def _extract_image_from_sample(self, sample: Dict) -> Optional[Image.Image]:
        """Extracts the PIL image from a sample dict using the 'jpg' key."""
        if 'jpg' in sample:
            img_key = 'jpg'
        elif 'output_image' in sample:
            img_key = 'output_image'
        else:
            return None
        
        image_data = sample[img_key]
        if isinstance(image_data, Image.Image):
            return image_data
        if isinstance(image_data, (bytes, bytearray)):
            try:
                return Image.open(io.BytesIO(image_data))
            except Exception:
                return None # Return None if bytes are not a valid image
        return None

    def _extract_caption_from_sample(self, sample: Dict) -> str:
        """Extracts the caption from a sample dict using the 'txt' key."""
        if 'txt' in sample:
            text_key = 'txt'
        elif "input_prompt" in sample:
            text_key = 'input_prompt'
        else:
            return "" # Return an empty string if caption is missing
        
        caption_data = sample[text_key]
        if isinstance(caption_data, (bytes, bytearray)):
            return caption_data.decode("utf-8", errors="ignore")
        if isinstance(caption_data, str):
            return caption_data
        return str(caption_data)

    def _process_pil(self, pil_image: Image.Image):
        """
        Processes a PIL image: converts to RGB, resizes, crops, and normalizes.
        Returns both the normalized tensor and the pre-normalization tensor.
        """
        pil_image = pil_image.convert('RGB')
        
        # Skip images that are smaller than the target resolution
        if min(*pil_image.size) < self.resolution:
            return None
            
        # Resize, crop, and convert to tensor
        pil_image = resize(pil_image, self.resolution)
        pil_image = self.crop_fn(pil_image, self.resolution, self.resolution)
        raw_image_tensor = to_tensor(pil_image)
        normalized_image_tensor = self.normalize(raw_image_tensor)
        
        return normalized_image_tensor, raw_image_tensor

    def _make_pipeline(self, worker_id: int, num_workers: int):
        """
        Creates the data loading pipeline using webdataset.
        This pipeline is optimized for performance in a multi-worker setup.
        """
        # `resampled=self.repeat` is a robust way to handle repeating datasets
        # and shuffling the order of shards for each epoch.
        handler = wds.warn_and_continue
        dataset = wds.DataPipeline(
            wds.SimpleShardList(self.urls),
            wds.shuffle(100),
            wds.split_by_node,
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(),
            # this shuffles the samples in memory
            wds.shuffle(self.shuffle_buffer),
            # this decodes the images and json
            wds.decode("pil", handler=handler),
        )
                    
        return dataset

    def __iter__(self):
        """The iterator method that yields data samples."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process data loading
            worker_id, num_workers = 0, 1
        else:
            # Multi-process data loading
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Create a unique pipeline for each worker
        pipeline = self._make_pipeline(worker_id, num_workers)

        for sample in pipeline:
            try:
                pil_image = self._extract_image_from_sample(sample)
                if pil_image is None:
                    print("skip image")
                    continue # Skip if sample has no valid image

                processed_data = self._process_pil(pil_image)
                if processed_data is None:
                    continue # Skip if image processing fails (e.g., too small)
                
                img_tensor, raw_image = processed_data
                caption = self._extract_caption_from_sample(sample)
                
                # Optional: Add a random prefix to short captions for augmentation
                if random.random() < 0.5 and len(caption.split()) < 30:
                    caption = random.choice(self.prefix_template) + caption

                metadata = {
                    "raw_image": raw_image, # Tensor before normalization
                    "prompt": caption,
                }
                yield (img_tensor, caption, metadata)
                
            except GeneratorExit:
                # This exception is raised when the consumer of the iterator stops.
                raise
            except Exception:
                # Catch-all for any other errors in a sample to make the stream robust.
                # In a real-world scenario, you might want to log this error.
                # e.g., print(f"Warning: Skipping a bad sample due to {e}")
                print("fail")
                continue
