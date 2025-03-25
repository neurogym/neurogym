import torch
from torch import nn

import numpy as np

from neurogym.wrappers.monitors.layers.base import LayerMonitorBase


class LinearLayerMonitor(LayerMonitorBase):

    layer_types: nn.Module | list[nn.Module] = nn.Linear

    def __init__(
        self,
        layer: nn.Linear,
        *args,
        **kwargs,
    ):

        self.layer: nn.Linear
        super().__init__(layer, *args, **kwargs)

    def _get_neuron_count(self) -> int:
        """Get the neuron count for this layer.

        Returns:
            int:
                The number of neurons.
        """
        return self.layer.out_features  # type: ignore[no-any-return]


    def _get_layer_output(
        self,
        output: torch.Tensor | tuple[torch.Tensor],
    ) -> np.ndarray:
        """Get the output for this layer.

        The output has a batch of 1, so return the only batch element as a NumPy array.

        Returns:
            np.ndarray:
                A NumPy array representing the output of this layer.
        """

        return output[0].detach().clone().cpu().numpy().squeeze()
