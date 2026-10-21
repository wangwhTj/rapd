import torch
import json
import copy
import os
from torch.utils.data import Dataset
from PIL import Image

def dpg_save_fn(image, metadata, root_path):
    image_path = os.path.join(root_path, str(metadata['filename'])+"_"+str(metadata['seed'])+".png")
    Image.fromarray(image).save(image_path)

class DPGDataset(Dataset):
    def __init__(self, prompt_path, num_samples_per_instance, latent_shape, seed=None, seeds=None):
        if seed is not None and seeds is not None:
            raise ValueError("Only one of seed or seeds can be set")
        self.latent_shape = latent_shape
        self.prompt_path = prompt_path
        prompt_files = sorted(
            prompt_file
            for prompt_file in os.listdir(self.prompt_path)
            if prompt_file.endswith(".txt")
        )
        self.prompts = []
        self.filenames = []
        for prompt_file in prompt_files:
            with open(os.path.join(self.prompt_path, prompt_file)) as fp:
                self.prompts.append(fp.readline().strip())
                self.filenames.append(prompt_file.replace('.txt', ''))
        self.num_instances = len(self.prompts)
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
        seed = self.seeds[sample_idx]
        generator = torch.Generator().manual_seed(seed)
        metadata = dict(
            prompt=self.prompts[instance_idx],
            filename=self.filenames[instance_idx],
            seed=seed,
            save_fn=dpg_save_fn,
        )
        condition = metadata["prompt"]
        latent = torch.randn(self.latent_shape, generator=generator, dtype=torch.float32)
        return latent, condition, metadata
