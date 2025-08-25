from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import numpy as np
    import torch
    from torch import nn


class LayerProbeBase(ABC):
    layer_type: type[nn.Module]
    probes: ClassVar

    @staticmethod
    def get_monitor(
        layer: nn.Module,
        hook: Callable,
        populations: set[str] | None = None,
    ) -> LayerProbeBase:
        """Mapping from Torch layer type to layer monitor.

        Args:
            layer: The layer being monitored.
            hook: A function to register as a forward hook.
            populations: Neuron populations to monitor.

        Returns:
            A dictionary mapping layer types to layer monitors.

        Raises:
            NotImplementedError: Raised if a monitor for this layer type has not been implemented.
        """
        cls = layer.__class__
        for layer_monitor in LayerProbeBase.__subclasses__():
            layer_type = layer_monitor.layer_type
            if cls is layer_type:
                return layer_monitor(layer, hook, populations=populations)  # type: ignore[abstract]

        msg = f"A monitor for layers of type '{cls}' has not been implemented yet."
        raise NotImplementedError(msg)

    def __init__(
        self,
        layer: nn.Module,
        hook: Callable,
        populations: str | Iterable | None,
    ):
        self.layer = layer

        layer_cls = self.__class__

        if layer_cls.probes is None:
            layer_cls.probes = {}

        if populations is None:
            populations = set(layer_cls.probes.keys())
        elif isinstance(populations, str):
            populations = {populations}
        else:
            populations = set(populations)

        self.populations = populations.intersection(set(layer_cls.probes.keys()))

        # Register a forward hook with the monitored layer.
        self.layer.register_forward_hook(hook, always_call=True)

    @abstractmethod
    def get_population_shapes(self) -> dict[str, tuple]:
        """Get the shape of the output of this layer.

        This should be overridden in derived classes.

        Returns:
            A tuple representing the shape of the layer's output.
        """

    def get_activations(
        self,
        output: torch.Tensor | tuple[torch.Tensor],
    ) -> dict[str, np.ndarray]:
        """Get the output for this layer.

        Potentially unpack a tuple of tensors, such as in the case of LSTM.
        This should be overridden in derived classes.

        Returns:
            A tensor representing the output of this layer.
        """
        return {probe: self.probes[probe](output) for probe in self.populations}

    @staticmethod
    def to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """A convenience method for converting a tensor into a NumPy array.

        Args:
            tensor: A PyTorch tensor

        Returns:
            A NumPy array.
        """
        return tensor.detach().clone().squeeze().cpu().numpy()
