import torch
import json
import copy
from torch.utils.data import Dataset
import os
from PIL import Image

def geneval_save_fn(image, metadata, root_path):
    path = os.path.join(root_path, metadata['filename'])
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    # save image
    image_path = os.path.join(path, "samples", f"{metadata['seed']}.png")
    if not os.path.exists(os.path.dirname(image_path)):
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
    Image.fromarray(image).save(image_path)
    # metadata_path
    metadata_path = os.path.join(path, "metadata.jsonl")
    with open(metadata_path, "w") as fp:
        json.dump(metadata, fp)

class GenEvalDataset(Dataset):
    def __init__(self, meta_json_path, num_samples_per_instance, latent_shape, seed=None, seeds=None, max_instances=None):
        if seed is not None and seeds is not None:
            raise ValueError("Only one of seed or seeds can be set")
        self.latent_shape = latent_shape
        self.meta_json_path = meta_json_path
        with open(meta_json_path) as fp:
            self.metadatas = [json.loads(line) for line in fp]
        if max_instances is not None:
            self.metadatas = self.metadatas[:int(max_instances)]
        self.num_instances = len(self.metadatas)
        self.seeds = self._make_seeds(num_samples_per_instance, seed, seeds)
        self.num_samples_per_instance = len(self.seeds)
        self.num_samples = self.num_instances * self.num_samples_per_instance

    @staticmethod
    def _make_seeds(num_samples_per_instance, seed, seeds):
        if seeds is not None:
            if isinstance(seeds, (int, str)):
                seeds = [seeds]
            seeds = [int(s) for s in seeds]
            if len(seeds) != int(num_samples_per_instance):
                raise ValueError(
                    "len(seeds) must equal num_samples_per_instance "
                    f"({len(seeds)} != {num_samples_per_instance})"
                )
            return seeds
        if seed is None:
            seed = 0
        seed = int(seed)
        return [seed + i for i in range(int(num_samples_per_instance))]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        instance_idx = idx // self.num_samples_per_instance
        sample_idx = idx % self.num_samples_per_instance
        metadata = copy.deepcopy(self.metadatas[instance_idx])
        seed = self.seeds[sample_idx]
        generator = torch.Generator().manual_seed(seed)
        condition = metadata["prompt"]
        latent = torch.randn(self.latent_shape, generator=generator, dtype=torch.float32)
        filename = f"{idx}"
        metadata["seed"] = seed
        metadata["filename"] = filename
        metadata["save_fn"] = geneval_save_fn
        return latent, condition, metadata
