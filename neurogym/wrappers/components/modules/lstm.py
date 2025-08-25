from typing import ClassVar

from torch import nn

from neurogym.wrappers.components.modules.base import LayerProbeBase


class LSTMLayerProbe(LayerProbeBase):
    layer_type = nn.LSTM
    probes: ClassVar = {
        "output": lambda output: LayerProbeBase.to_numpy(output[0]),
        "hidden": lambda output: LayerProbeBase.to_numpy(output[1][0]),
        "cell": lambda output: LayerProbeBase.to_numpy(output[1][1]),
    }

    def get_population_shapes(self) -> dict[str, tuple]:
        # TODO: Handle multi-layer and bidirectional LSTMs
        return {
            "output": (self.layer.hidden_size,),
            "hidden": (self.layer.hidden_size,),
            "cell": (self.layer.hidden_size,),
        }
