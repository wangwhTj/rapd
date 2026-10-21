import torch
import torch.nn as nn
from src.models.conditioner.base import BaseConditioner
from typing import List, Optional

from transformers import Qwen3Model, Qwen2Tokenizer


class Qwen3TextEncoder(BaseConditioner):
    def __init__(
        self,
        weight_path: str,
        embed_dim: int = None,
        max_length=128,
        enable_chi_prompt: bool = False,
        chi_prompt: Optional[List[str]] = None,
    ):
        super().__init__()
        self.tokenizer = Qwen2Tokenizer.from_pretrained(weight_path, max_length=max_length, padding_side="right")
        # self.model = Qwen3Model.from_pretrained(weight_path, attn_implementation="flex_attention").to(torch.bfloat16)
        self.model = Qwen3Model.from_pretrained(weight_path).to(torch.bfloat16)
        self.model.compile()
        self.uncondition_embedding = None
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.enable_chi_prompt = enable_chi_prompt
        self.chi_prompt = "\n".join(chi_prompt or [])
        self.num_chi_prompt_tokens = len(self.tokenizer.encode(self.chi_prompt)) if self.chi_prompt else 0
        # torch._dynamo.config.optimize_ddp = False

    def _fit_embed_dim(self, y):
        if self.embed_dim is None:
            return y
        if y.shape[2] < self.embed_dim:
            y = torch.cat([y, torch.zeros(y.shape[0], y.shape[1], self.embed_dim - y.shape[2]).to(y.device, y.dtype)], dim=-1)
        if y.shape[2] > self.embed_dim:
            y = y[:, :, :self.embed_dim]
        return y

    def _prepare_prompts(self, y, apply_chi_prompt: bool):
        if not (apply_chi_prompt and self.enable_chi_prompt and self.chi_prompt):
            return y, self.max_length, None

        prompts = [self.chi_prompt + prompt for prompt in y]
        max_length_all = self.num_chi_prompt_tokens + self.max_length
        select_index = list(range(-self.max_length, 0))
        return prompts, max_length_all, select_index

    def _impl_condition(self, y, metadata:dict=None, apply_chi_prompt: bool = True):
        if metadata is None:
            metadata = {}
        y, max_length, select_index = self._prepare_prompts(y, apply_chi_prompt)
        tokenized = self.tokenizer(y, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
        input_ids = tokenized.input_ids.cuda()
        attention_mask = tokenized.attention_mask.cuda()
        y = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        if select_index is not None:
            y = y[:, select_index]
            attention_mask = attention_mask[:, select_index]
        metadata["valid_length_y"] = torch.sum(attention_mask, dim=-1)
        return self._fit_embed_dim(y)

    def _impl_uncondition(self, y, metadata:dict=None):
        if metadata is None:
            metadata = {}
        if self.uncondition_embedding is not None and "negative_prompt" not in metadata:
            return self.uncondition_embedding.repeat(len(y), 1, 1)
        negative_prompt = "" if "negative_prompt" not in metadata else metadata['negative_prompt']
        self.uncondition_embedding = self._impl_condition([negative_prompt,], apply_chi_prompt=False)
        return self.uncondition_embedding.repeat(len(y), 1, 1)
