from typing import Any, Dict

import torch
import torch.nn as nn


class SimpleEMA:
    def __init__(self,
                 decay: float = 0.9999,
                 every_n_steps: int = 1,
                 ):
        self.decay = decay
        self.every_n_steps = every_n_steps
        self._stream = torch.cuda.Stream()
        self.previous_step = 0

    def setup_models(self, net: nn.Module, ema_net: nn.Module):
        self.net_params = list(net.parameters())
        self.ema_params = list(ema_net.parameters())

    def ema_step(self):
        @torch.no_grad()
        def ema_update(ema_model_tuple, current_model_tuple, decay):
            torch._foreach_mul_(ema_model_tuple, decay)
            torch._foreach_add_(
                ema_model_tuple, current_model_tuple, alpha=(1.0 - decay),
            )

        if self._stream is not None:
            self._stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._stream):
            ema_update(self.ema_params, self.net_params, self.decay)
        assert self.ema_params[0].dtype == torch.float32

    def synchronize(self):
        if self._stream is not None:
            self._stream.synchronize()

    def step(self, global_step: int) -> None:
        if global_step == self.previous_step:
            return
        self.previous_step = global_step
        if global_step % self.every_n_steps == 0:
            self.ema_step()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "every_n_steps": self.every_n_steps,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.decay = state_dict["decay"]
        self.every_n_steps = state_dict["every_n_steps"]
