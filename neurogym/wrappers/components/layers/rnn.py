from typing import ClassVar

from torch import nn

from neurogym.wrappers.components.layers.base import LayerMonitorBase, to_numpy


class RNNLayerMonitor(LayerMonitorBase):
    layer_type = nn.RNN
    probes: ClassVar = {
        "output": lambda output: to_numpy(output[0]),
        "hidden": lambda output: to_numpy(output[1]),
    }

    def get_population_shapes(self) -> dict[str, tuple]:
        # TODO: Handle multi-layer and bidirectional RNNs
        return {
            "output": (self.layer.hidden_size,),
            "hidden": (self.layer.hidden_size,),
        }
