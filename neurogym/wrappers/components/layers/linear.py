import numpy as np
import torch
from torch import nn

from neurogym.wrappers.components.layers.base import LayerMonitorBase


class LinearLayerMonitor(LayerMonitorBase):
    supported = nn.Linear

    def get_output_shape(self) -> tuple[int, ...]:
        """Get the shape of the output of this layer.

        This should be overridden in derived classes.

        Returns:
            A tuple representing the shape of the layer's output.
        """
        return (self.layer.out_features,)  # type: ignore[no-any-return]

    def get_output(
        self,
        output: torch.Tensor | tuple[torch.Tensor],
    ) -> np.ndarray:
        """Get the output for this layer.

        The output has a batch of 1, so return the only batch element as a NumPy array.

        Returns:
            A NumPy array representing the output of this layer.
        """
        return output[0].detach().clone().cpu().numpy().squeeze()
