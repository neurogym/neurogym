# --------------------------------------
import numpy as np

# --------------------------------------
import torch
from torch import nn

# --------------------------------------
from bokeh.models import Tabs
from bokeh.models import TabPanel

# --------------------------------------
from neurogym.config.components.monitor import NetParam
from neurogym.wrappers.bokehmon.components.base import ComponentBase
from neurogym.wrappers.bokehmon.components.activation import ActivationComponent


class LayerBase:

    def __init__(
        self,
        module: nn.Module,
        params: list[NetParam],
        name: str,
    ):
        """
        A base class for all types of layer monitors.

        Args:
            module (nn.Module):
                The module being monitored.

            params (list[NetParam]):
                A list of parameter types to keep track of.

            name (str):
                The name of the layer.
        """

        # Store the parameters
        self.module = module
        self.params = params
        self.name = name

        # Data
        # ==================================================
        self.trial = 0
        self.step = 0

        # Components
        # ==================================================
        self.components: dict[NetParam, ComponentBase] = {}
        self._register_components()

        # Register the hooks for this module
        # ==================================================
        self.module.register_forward_hook(self._forward_hook, always_call=True)

    def _register_components(self):
        """
        Register monitoring components based on the requested parameter types.
        """

        components = {
            NetParam.Activation: ActivationComponent,
            NetParam.Weight: None,  # TODO
        }

        if self.module.bias is not None:
            components[NetParam.Bias] = None  # TODO

        for param in self.params:
            component = components.get(param, None)
            if component is not None:
                self.components[param] = component(self.module)

    def _forward_hook(
        self,
        module: nn.Module,
        input_: torch.Tensor,
        output: torch.Tensor,
    ):
        """
        A forward hook.

        Args:
            module (nn.Module):
                The module that this hook is registered with.

            input_ (torch.Tensor):
                The current input provided to the module.

            output (torch.Tensor):
                The module's output.
        """

        # Compute the batch mean
        batch_mean_output = np.mean(output.detach().clone().numpy(), axis=0)

        for param, component in self.components.items():
            component.process(module, input_, batch_mean_output, self.trial, self.step)

        self.step += 1

    def _get_neuron_count(self) -> int:
        """
        Get the neuron count for this layer.
        This should be overridden in derived classes.

        Returns:
            int:
                The number of neurons.
        """
        raise NotImplementedError("Please implement this method in a derived class.")

    def _get_channel_count(self) -> int:
        """
        Get the channel count for this layer.
        This should be overridden in derived classes.

        Returns:
            int:
                The number of channels.
        """
        raise NotImplementedError("Please implement this method in a derived class.")

    def _plot(self) -> TabPanel:
        """
        Render this layer as a tab, with subtabs for each component.

        Returns:
            TabPanel:
                The components for this layer encapsulated in a tab.
        """

        return TabPanel(
            title=self.name,
            child=Tabs(
                tabs=[component._plot() for component in self.components.values()]
            ),
        )

    def _start_trial(self):
        """
        Start monitoring parameters for a new trial.
        """

        for param, component in self.components.items():
            component._start_trial()

        self.trial += 1
        self.step = 0
