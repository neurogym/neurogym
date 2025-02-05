from typing import Any

import torch
from bokeh.models import TabPanel
from torch import nn


class ComponentBase:
    def __init__(
        self,
        layer: Any,
        module: nn.Module,
    ):
        """A base class for all monitor components.

        Args:
            layer (Any):
                The parent layer that this component is attached to.

            module (nn.Module):
                The module being monitored.
        """
        self.layer = layer
        self.module = module

    def get_neuron_count(self) -> int:
        """Get the total neuron count for the parent layer.

        Returns:
            int:
                The number of neurons in the layer.
        """
        return self.layer.get_neuron_count()  # type: ignore[no-any-return]

    def get_channel_count(self) -> int:
        """Get the channel count for the parent layer.

        Returns:
            int:
                The number of channels in the layer.
        """
        return self.layer.get_channel_count()  # type: ignore[no-any-return]

    def process(
        self,
        module: nn.Module,
        input_: torch.Tensor,
        output: torch.Tensor,
        trial: int,
        step: int,
    ):
        """Process data from a module hook.

        This method takes data provided by a module hook
        for the current step in the current trial and optionally
        stores and processes the data.

        Args:
            module (nn.Module):
                The module that this hook is registered with.

            input_ (torch.Tensor):
                The current input provided to the module.

            output (torch.Tensor):
                The module's output.

            trial (int):
                The current trial.

            step (int):
                The current step in the current trial.

        Raises:
            NotImplementedError:
                Raised when trying to use the base class directly.
        """
        msg = "Please implement this method in a derived class."
        raise NotImplementedError(msg)

    def start_trial(self):
        """Start monitoring parameters for a new trial."""

    def plot(self) -> TabPanel:
        """Render this component as a tab.

        Raises:
            NotImplementedError:
                Raised when trying to use the base class directly.

        Returns:
            TabPanel:
                The current component as a tab.
        """
        msg = "Please implement this method in a derived class."
        raise NotImplementedError(msg)
