from torch import nn
from neurogym.wrappers.components.layers.base import to_numpy
from neurogym.wrappers.components.layers.base import LayerMonitorBase


class LSTMLayerMonitor(LayerMonitorBase):
    layer_types: nn.Module = nn.LSTM
    probes: dict = {
        "output": lambda output: to_numpy(output[0]),
        "hidden": lambda output: to_numpy(output[1][0]),
        "cell": lambda output: to_numpy(output[1][1]),
    }

    def get_population_shapes(self) -> dict[str, tuple]:
        # TODO: Handle multi-layer and bidirectional LSTMs
        return {
            "output": (self.layer.hidden_size,),
            "hidden": (self.layer.hidden_size,),
            "cell": (self.layer.hidden_size,),
        }
