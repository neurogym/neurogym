from torch import nn
from neurogym.wrappers.components.layers.base import to_numpy
from neurogym.wrappers.components.layers.base import LayerMonitorBase


class LinearLayerMonitor(LayerMonitorBase):
    layer_types = nn.Linear
    probes = {"output": lambda output: to_numpy(output)}

    def get_population_shapes(self) -> dict[str, tuple]:
        """Get the shape of the output of this layer.

        Returns:
            A tuple representing the shape of the layer's output.
        """
        return {"output": (self.layer.out_features,)}
