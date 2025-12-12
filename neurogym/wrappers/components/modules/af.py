from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from neurogym.wrappers.components.modules.base import LayerProbeBase


@dataclass
class AFLayerProbe(LayerProbeBase):
    @staticmethod
    def layer_types() -> set[type[nn.Module]]:
        """The layer types that this probe applies to."""
        return {
            nn.ELU,
            nn.Hardshrink,
            nn.Hardsigmoid,
            nn.Hardtanh,
            nn.Hardswish,
            nn.LeakyReLU,
            nn.LogSigmoid,
            nn.MultiheadAttention,
            nn.PReLU,
            nn.ReLU,
            nn.ReLU6,
            nn.RReLU,
            nn.SELU,
            nn.CELU,
            nn.GELU,
            nn.Sigmoid,
            nn.SiLU,
            nn.Mish,
            nn.Softplus,
            nn.Softshrink,
            nn.Softsign,
            nn.Tanh,
            nn.Tanhshrink,
            nn.Threshold,
            nn.GLU,
            nn.Softmin,
            nn.Softmax,
            nn.Softmax2d,
            nn.LogSoftmax,
            nn.AdaptiveLogSoftmaxWithLoss,
        }

    def _setup_probes(self) -> dict[str, Callable]:
        """Set up the probes for this type of layer."""
        return {"output": self._get_output_activations}

    def _get_output_activations(self, tensor: torch.Tensor) -> np.ndarray:
        """Return the activations of the output neurons."""
        return self.to_numpy(tensor)
