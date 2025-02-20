from __future__ import annotations

from typing import Any

import torch
from bokeh.models import TabPanel
from torch import nn

import neurogym as ngym


class ParamMonitorBase:

    param_type: ngym.NetParam = None

    @staticmethod
    def param_monitor_types() -> dict[ngym.NetParam, ParamMonitorBase]:
        """Mapping from ngym.NetParam type to parameter monitor type.

        Returns:
            dict[ngym.NetParam, ParamMonitorBase]:
                A dictionary mapping layer types to layer monitors.
        """
        return {
            param_monitor.param_type: param_monitor
            for param_monitor in ParamMonitorBase.__subclasses__()
        }

    def __init__(
        self,
        monitor: Any,
        layer: nn.Module,
        phases: set[ngym.MonitorPhase],
    ):
        """A base class for all monitor components.

        Args:
            monitor (Any):
                The layer monitor that this component is attached to.

            layer (nn.Module):
                The layer being monitored.

            phases (set[ngym.MonitorPhase]):
                Phases during which the parameters should be monitored.
        """
        self.monitor = monitor
        self.layer = layer
        self.phases = phases
        self.phase: ngym.MonitorPhase = None

    def _init_containers(self):
        """Initialise the data container(s).

        This cannot and should not be implemented here because
        each parameter monitor would likely have a different
        storage structure and mechanisms for modifying it.

        Raises:
            NotImplementedError:
                Raised when trying to use the base class directly.
        """
        msg = "Please implement this method in a derived class."
        raise NotImplementedError(msg)

    def get_neuron_count(self) -> int:
        """Get the total neuron count for the parent layer.

        Returns:
            int:
                The number of neurons in the layer.
        """
        return self.monitor.get_neuron_count()  # type: ignore[no-any-return]

    def get_channel_count(self) -> int:
        """Get the channel count for the parent layer.

        Returns:
            int:
                The number of channels in the layer.
        """
        return self.monitor.get_channel_count()  # type: ignore[no-any-return]

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

        Raises:
            NotImplementedError:
                Raised when trying to use the base class directly.
        """
        msg = "Please implement this method in a derived class."
        raise NotImplementedError(msg)

    def start_trial(self):
        """Start monitoring parameters for a new trial.
        Raises:
            NotImplementedError:
                Raised when trying to use the base class directly.
        """

        msg = "Please implement this method in a derived class."
        raise NotImplementedError(msg)

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
