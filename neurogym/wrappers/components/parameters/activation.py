import numpy as np
import torch
from torch import nn

from neurogym.wrappers.components.layers.base import LayerMonitorBase


class ActivationMonitor:
    def __init__(
        self,
        module: nn.Module,
        steps: int,
        name: str,
    ):
        """Activation monitoring component.

        Args:
            module: The layer being monitored.
            steps: The steps that this layer should be monitored for.
            name: The name of this monitor.
        """
        self.steps: int = steps
        self.monitor = LayerMonitorBase.get_monitor(module, self._fw_activation_hook)
        self.name = name

        # The neuron count
        self.paused = False
        self.history: list[np.ndarray] = []
        self.step: int = 0
        self._new_trial: bool = False
        self.start_new_trial()

    @property
    def new_trial(self) -> bool:
        """Get the _new_trial attribute."""
        return self._new_trial

    @new_trial.setter
    def new_trial(self, value: bool):
        """Set the _new_trial attribute."""
        self._new_trial = value

    def _fw_activation_hook(
        self,
        module: nn.Module,  # noqa: ARG002
        args: torch.Tensor,  # noqa: ARG002
        output: torch.Tensor,
    ):
        """A forward hook.

        Args:
            module: The module that this hook is registered with.
            args: Function arguments.
            output: The module's output.
        """
        # Only do something if recording is not paused and we are below the requested number of steps
        if self._new_trial:
            self.start_new_trial()

        if (not self.paused) and (self.step < self.steps):
            # Extract the relevant output for this layer and store it.
            self.history[-1][self.step] = self.monitor.get_output(output)
            self.step += 1

    def start_new_trial(self):
        """Start recording activations for a new trial."""
        self.history.append(np.ma.masked_all((self.steps, *self.monitor.get_output_shape())))
        self.step = 0
        self._new_trial = False

    def pause(self):
        """Pause recording."""
        self.paused = True

    def resume(self):
        """Resume recording."""
        self.paused = False
