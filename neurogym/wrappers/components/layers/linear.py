from typing import ClassVar

from torch import nn

from neurogym.wrappers.components.layers.base import LayerMonitorBase, to_numpy


class LinearLayerMonitor(LayerMonitorBase):
    layer_type = nn.Linear
    probes: ClassVar = {"output": lambda output: to_numpy(output)}

    def get_population_shapes(self) -> dict[str, tuple]:
        """Get the shape of the output of this layer.

        Returns:
            A tuple representing the shape of the layer's output.
        """
        return {"output": (self.layer.out_features,)}
