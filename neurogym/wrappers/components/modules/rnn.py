from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from neurogym.wrappers.components.modules.base import LayerProbeBase


@dataclass
class RNNLayerProbe(LayerProbeBase):
    layer: nn.RNN

    @staticmethod
    def layer_types() -> set[type[nn.Module]]:
        """The layer types that this probe applies to."""
        return {nn.RNN}

    def _setup_probes(self) -> dict[str, Callable]:
        """Set up the probes for this type of layer."""
        return {
            "output": self._get_output_activations,
            "hidden": self._get_hidden_activations,
        }

    def get_population_shapes(self) -> dict[str, tuple]:
        """Get the shape of the output of this layer."""
        return {
            "output": (self.layer.hidden_size,),
            "hidden": (self.layer.hidden_size,),
        }

    def _get_output_activations(self, tensor: torch.Tensor) -> np.ndarray:
        """Return the activations of the output neurons."""
        return self.to_numpy(tensor[0])

    def _get_hidden_activations(self, tensor: torch.Tensor) -> np.ndarray:
        """Return the activations of the hidden neurons."""
        return self.to_numpy(tensor[1])
