from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np


class LayerMonitorBase(ABC):
    supported = nn.Module | None

    @staticmethod
    def get_monitor(layer: nn.Module, hook: Callable) -> LayerMonitorBase:
        """Mapping from Torch layer type to layer monitor.

        Args:
            layer: The layer being monitored.
            hook: A function to register as a forward hook.

        Returns:
            A dictionary mapping layer types to layer monitors.

        Raises:
            NotImplementedError: Raised if a monitor for this layer type has not been implemented.
        """
        cls = layer.__class__
        for layer_monitor in LayerMonitorBase.__subclasses__():
            if cls is layer_monitor.supported:
                return layer_monitor(layer, hook)  # type: ignore[abstract]

        msg = f"It seems that a monitor for layers of type '{cls}' has not been implemented yet."
        raise NotImplementedError(msg)

    def __init__(
        self,
        layer: nn.Module,
        hook: Callable,
    ):
        """A base class for all types of layer monitors.

        Args:
            layer: The layer being monitored.
            hook: A function to register as a forward hook.
        """
        self.layer = layer

        # Register a forward hook with the monitored layer.
        # ==================================================
        self.layer.register_forward_hook(hook, always_call=True)

    @abstractmethod
    def get_output_shape(self) -> tuple[int, ...]:
        """Get the shape of the output of this layer.

        This should be overridden in derived classes.

        Returns:
            A tuple representing the shape of the layer's output.
        """

    @abstractmethod
    def get_output(
        self,
        output: torch.Tensor | tuple[torch.Tensor],
    ) -> np.ndarray:
        """Get the output for this layer.

        Potentially unpack a tuple of tensors, such as in the case of LSTM.
        This should be overridden in derived classes.

        Returns:
            A tensor representing the output of this layer.
        """
