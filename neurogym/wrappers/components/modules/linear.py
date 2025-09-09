from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from neurogym.wrappers.components.modules.base import LayerProbeBase


@dataclass
class LinearLayerProbe(LayerProbeBase):
    layer: nn.Linear

    @staticmethod
    def layer_types() -> set[type[nn.Module]]:
        """The layer types that this probe applies to."""
        return {nn.Linear}

    def _setup_probes(self) -> dict[str, Callable]:
        """Set up the probes for this type of layer."""
        return {"output": self._get_output_activations}

    def get_population_shapes(self) -> dict[str, tuple]:
        """Get the shape of the output of this layer."""
        return {"output": (self.layer.out_features,)}

    def _get_output_activations(self, tensor: torch.Tensor) -> np.ndarray:
        """Return the activations of the output neurons."""
        return self.to_numpy(tensor)
