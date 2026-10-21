"""Frozen DINOv2 feature extraction for diffusion feature alignment."""
import copy
import os
import torch
import torch.nn as nn
from torchvision.transforms import Normalize
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class DINOv2(nn.Module):
    def __init__(self, weight_path:str=None, base_patch_size=16):
        super(DINOv2, self).__init__()
        hub_dir = torch.hub.get_dir()
        local_repo = None
        if os.path.isdir(hub_dir):
            for name in os.listdir(hub_dir):
                if name.startswith("facebookresearch_dinov2"):
                    local_repo = os.path.join(hub_dir, name)
                    break
        if local_repo is not None and os.path.isdir(local_repo):
            self.encoder = torch.hub.load(local_repo, "dinov2_vitb14", source="local", trust_repo=True)
        else:
            self.encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True)

        if weight_path and os.path.isfile(weight_path):
            state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
            self.encoder.load_state_dict(state_dict, strict=False)

        self.pos_embed = copy.deepcopy(self.encoder.pos_embed)
        self.encoder.head = torch.nn.Identity()
        self.patch_size = self.encoder.patch_embed.patch_size
        self.precomputed_pos_embed = dict()
        self.base_patch_size = base_patch_size
        self.encoder.compile()

    def forward(self, x, resize=True):
        b, c, h, w = x.shape
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        if resize:
            x = torch.nn.functional.interpolate(x, (int(14*h/self.base_patch_size), int(14*w/self.base_patch_size)), mode='bicubic')
        feature = self.encoder.forward_features(x)['x_norm_patchtokens']
        return feature
    
    @torch.compile
    def get_intermediate_feats(self, x, resize=True, n=None, reshape=False, return_class_token=False):
        if n is None:
            n = [2, 5, 8, 11]
        b, c, h, w = x.shape
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        
        if resize:
            target_h = int(14 * h / self.base_patch_size)
            target_w = int(14 * w / self.base_patch_size)
            x = torch.nn.functional.interpolate(x, (target_h, target_w), mode='bicubic')

        features = self.encoder.get_intermediate_layers(x, n=n, reshape=reshape, return_class_token=return_class_token)
        return features
    
    def forward_with_cls(self, x, resize=True):
        b, c, h, w = x.shape
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        if resize:
            x = torch.nn.functional.interpolate(x, (int(14*h/self.base_patch_size), int(14*w/self.base_patch_size)), mode='bicubic')
        out = self.encoder.forward_features(x)
        feature, cls_token = out['x_norm_patchtokens'], out["x_norm_clstoken"].unsqueeze(1)
        return feature, cls_token
