from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    import torch
    from torch import nn


@dataclass
class LayerProbeBase:
    layer: nn.Module
    hook: Callable
    populations: set[str]
    probes: dict = field(default_factory=dict)

    @staticmethod
    def layer_types() -> set[type[nn.Module]]:
        """The layer types that this probe applies to."""
        return set()

    @staticmethod
    def get_probe(
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
            if cls in layer_monitor.layer_types():
                return layer_monitor(layer, hook, populations or set())

        msg = f"A monitor for layers of type '{cls}' has not been implemented yet."
        raise NotImplementedError(msg)

    def __post_init__(self):
        self.probes = self._setup_probes()
        if isinstance(self.populations, str):
            self.populations = {self.populations}

        if not self.populations:
            self.populations = set(self.probes)

        self.populations = self.populations.intersection(set(self.probes.keys()))

        # Register a forward hook with the monitored layer.
        self.layer.register_forward_hook(self.hook, always_call=True)

    def _setup_probes(self) -> dict[str, Callable]:
        """Set up the probes for this type of layer."""
        return {}

    def get_population_shapes(self) -> dict[str, tuple]:
        """Get the shape of the output of this layer.

        Returns:
            A dictionary representing the shape of the layer's output(s).
        """
        return {}

    def get_activations(
        self,
        output: torch.Tensor | tuple[torch.Tensor],
    ) -> dict[str, np.ndarray]:
        """Get the output for this layer.

        This functionality makes use of dispatch functions,
        which should be aware of any idiosyncrasies of the format of the output tensor.
        For instance, the output of an LSTM layer is of the form `(output, (hidden, cell))`.

        Args:
            output: One or more tensors output by the layer.

        Returns:
            A tensor representing the output of this layer for all recorded populations.
        """
        return {probe: self.probes[probe](output) for probe in self.populations}

    def to_numpy(self, tensor: torch.Tensor, squeeze: tuple[int] | None = None) -> np.ndarray:
        """A convenience method for converting a tensor into a NumPy array.

        Args:
            tensor: A PyTorch tensor.
            squeeze: Dimensions to squeeze.

        Returns:
            A NumPy array.
        """
        _tensor = tensor.detach().clone()
        if squeeze is not None:
            _tensor.squeeze_(*squeeze)

        return _tensor.cpu().numpy()
