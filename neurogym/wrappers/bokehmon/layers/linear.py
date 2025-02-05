from neurogym.wrappers.bokehmon.layers.base import LayerBase


class LinearLayerMonitor(LayerBase):
    def get_neuron_count(self) -> int:
        """Get the neuron count for this layer.

        Returns:
            int:
                The number of neurons.
        """
        return self.module.out_features  # type: ignore[no-any-return]
