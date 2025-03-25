import torch
from torch import nn

import numpy as np

import neurogym as ngym
from neurogym.wrappers.monitors.layers.base import LayerMonitorBase


class LSTMLayerMonitor(LayerMonitorBase):

    layer_types: nn.Module | list[nn.Module] = nn.modules.rnn.LSTM

    def __init__(
        self,
        layer: nn.modules.rnn.LSTM,
        *args,
        **kwargs,
    ):

        self.layer: nn.modules.rnn.LSTM
        super().__init__(layer, *args, **kwargs)

    def _get_neuron_count(self) -> int:
        """Get the neuron count for this layer.

        Returns:
            int:
                The number of neurons.
        """
        return self.layer.hidden_size  # type: ignore[no-any-return]

    def _get_layer_output(
        self,
        output: torch.Tensor,
    ) -> np.ndarray:
        """Get the output for this layer.

        The output of an LSTM cell consists of three tensors.
        Unpack it and return only the first batch element of
        the first tensor (the batch size is assumed to be 1).

        Returns:
            np.ndarray:
                A NumPy array representing the output of this layer.
        """

        return output[0][0].detach().clone().cpu().numpy().squeeze()
