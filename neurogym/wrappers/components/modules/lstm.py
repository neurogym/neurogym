from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from neurogym.wrappers.components.modules.base import LayerProbeBase


@dataclass
class LSTMLayerProbe(LayerProbeBase):
    layer: nn.LSTM

    @staticmethod
    def layer_types() -> set[type[nn.Module]]:
        """The layer types that this probe applies to."""
        return {nn.LSTM}

    def _setup_probes(self) -> dict[str, Callable]:
        """Set up the probes for this type of layer."""
        return {
            "output": self._get_output_activations,
            "hidden": self._get_hidden_activations,
            "cell": self._get_cell_activations,
        }

    def get_population_shapes(self) -> dict[str, tuple]:
        """Get the shape of the output of this layer."""
        return {
            "output": (self.layer.hidden_size,),
            "hidden": (self.layer.hidden_size,),
            "cell": (self.layer.hidden_size,),
        }

    def _squeeze_check(self, tensor: torch.Tensor) -> tuple[int] | None:
        if len(tensor.shape) == 3:
            return (1,)
        return None

    def _get_output_activations(self, tensor: torch.Tensor) -> np.ndarray:
        """Return the activations of the output neurons."""
        _tensor = tensor[0]
        return self.to_numpy(_tensor, squeeze=self._squeeze_check(_tensor))

    def _get_hidden_activations(self, tensor: torch.Tensor) -> np.ndarray:
        """Return the activations of the hidden neurons."""
        _tensor = tensor[1][0]
        return self.to_numpy(_tensor, squeeze=self._squeeze_check(_tensor))

    def _get_cell_activations(self, tensor: torch.Tensor) -> np.ndarray:
        """Return the activations of the cell neurons."""
        _tensor = tensor[1][1]
        return self.to_numpy(_tensor, squeeze=self._squeeze_check(_tensor))
