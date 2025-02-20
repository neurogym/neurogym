from torch import nn

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

    def get_neuron_count(self) -> int:
        """Get the neuron count for this layer.

        Returns:
            int:
                The number of neurons.
        """
        return self.layer.out_features  # type: ignore[no-any-return]
