from __future__ import annotations

import numpy as np
import torch
from bokeh.models import TabPanel, Tabs
from torch import nn

import panel as pn

import neurogym as ngym
from neurogym.wrappers.monitors.parameters.base import ParamMonitorBase


class LayerMonitorBase:

    layer_types: nn.Module | list[nn.Module] = nn.Module

    @staticmethod
    def layer_monitors() -> dict[nn.Module, LayerMonitorBase]:
        """Mapping from Torch layer type to layer monitor.

        Returns:
            dict[nn.Module, LayerMonitorBase]:
                A dictionary mapping layer types to layer monitors.
        """
        return {
            layer_monitor.layer_types: layer_monitor
            for layer_monitor in LayerMonitorBase.__subclasses__()
        }

    def __init__(
        self,
        layer: nn.Module,
        params: list[ngym.NetParam],
        phases: set[ngym.MonitorPhase],
        name: str,
    ):
        """A base class for all types of layer monitors.

        Args:
            layer (nn.Module):
                The layer being monitored.

            params (list[ngym.NetParam]):
                A list of parameter types to keep track of.

            phases (set[ngym.MonitorPhase]):
                Phases during which the parameters should be monitored.

            name (str):
                The name of the layer.
        """
        self.layer: nn.Module = layer
        self._check_layer_type()

        self.params: list[ngym.NetParam] = params
        self.phases: set[ngym.MonitorPhase] = phases
        self.phase: ngym.MonitorPhase = None
        self.name: str = name

        # Data
        # ==================================================
        self.trial = 0
        self.step = 0

        # Parameter monitors
        # ==================================================
        self.param_monitors: dict[ngym.NetParam, ParamMonitorBase] = {}
        self._register_param_monitors()

        # Register a forward hook with the monitored layer.
        # ==================================================
        self.layer.register_forward_hook(self._forward_hook, always_call=True)  # type: ignore[arg-type]

    def _check_layer_type(self):
        """Checks if the layer is of the supported type.

        Raises:
            TypeError:
                Raised if the wrong layer type is passed.
        """
        cls = self.__class__
        if not isinstance(self.layer, cls.layer_types):
            msg = f"Please use {cls.layer_types} layers with monitors of type {cls}."
            raise TypeError(msg)

    def _register_param_monitors(self):
        """Register monitoring components based on the requested parameter types."""

        available_monitors = ParamMonitorBase.param_monitor_types()

        for param in self.params:
            match (param):
                case ngym.NetParam.Bias:
                    # Skip the monitor if the layer doesn't have biases.
                    if self.layer.bias is None:
                        continue
                case _:
                    pass

            # Get the monitoring component for this parameter
            param_monitor = available_monitors.get(param)
            if param_monitor is not None:
                self.param_monitors[param] = param_monitor(
                    self, self.layer, self.phases
                )

    def _forward_hook(
        self,
        module: nn.Module,
        input_: torch.Tensor,
        output: torch.Tensor,
    ):
        """A forward hook.

        Args:
            module (nn.Module):
                The module that this hook is registered with.

            input_ (torch.Tensor):
                The current input provided to the module.

            output (torch.Tensor):
                The module's output.
        """
        # Only do something if we are in a monitoring phase.
        if self.phase in self.phases:

            # Extract the relevant output for this layer
            # since the output could consist of multiple tensors.
            output = self._get_layer_output(output)

            for monitor in self.param_monitors.values():
                monitor.process(module, input_, output)

        self.step += 1

    def _get_neuron_count(self) -> int:
        """Get the neuron count for this layer.

        This should be overridden in derived classes.

        Returns:
            int:
                The number of neurons.
        """
        msg = "Please implement this method in a derived class."
        raise NotImplementedError(msg)

    def _get_layer_output(
        self,
        output: torch.Tensor | tuple[torch.Tensor],
    ) -> torch.Tensor:
        """Get the output for this layer.

        Potentially unpack a tuple of tensors, such as in the case of LSTM.

        Returns:
            torch.Tensor:
                A single tensor representing the output of this layer.
        """
        msg = "Please implement this method in a derived class."
        raise NotImplementedError(msg)

    def _get_channel_count(self) -> int:
        """Get the channel count for this layer.

        This should be overridden in derived classes.

        Returns:
            int:
                The number of channels.
        """
        msg = "Please implement this method in a derived class."
        raise NotImplementedError(msg)

    def plot(self) -> TabPanel:
        """Render this layer as a tab, with subtabs for each component.

        Returns:
            TabPanel:
                The components for this layer encapsulated in a tab.
        """
        # return pn.Tabs(
        #     title=self.name,
        #     child=Tabs(
        #         tabs=[monitor.plot() for monitor in self.param_monitors.values()],
        #     ),
        # )
        return pn.Tabs(
            *[(name.capitalize(), mon.plot()) for name, mon in self.param_monitors.items()],
        )

    def start_new_trial(self):
        """Start monitoring parameters for a new trial."""
        for monitor in self.param_monitors.values():
            monitor.start_new_trial()

        self.trial += 1
        self.step = 0

    def set_phase(
        self,
        phase: ngym.MonitorPhase,
    ):
        """Set the current phase.

        This can be used to switch monitoring on and off.

        Args:
            phase (ngym.MonitorPhase):
                A MonitorPhase enum indicating a phase in the
                pipeline, such as training or evaluation.
        """
        self.phase = phase

        for monitor in self.param_monitors.values():
            monitor.set_phase(phase)
