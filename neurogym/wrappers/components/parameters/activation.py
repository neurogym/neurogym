import numpy as np
import torch
from torch import nn

from neurogym.wrappers.components.layers.base import LayerMonitorBase


class ActivationMonitor:
    def __init__(
        self, module: nn.Module, steps: int, name: str, populations: set[str] | None = None
    ):
        """Activation monitoring component.

        Args:
            module: The layer being monitored.
            steps: The steps that this layer should be monitored for.
            name: The name of this monitor.
            populations: Neuron populations to monitor.

        """
        self.steps: int = steps
        self.monitor = LayerMonitorBase.get_monitor(
            module, self._fw_activation_hook, populations
        )
        self.name = name

        # The neuron count
        self.paused = False
        self.history: list[np.ndarray] = {
            population: [] for population in self.monitor.populations
        }
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
        input_: torch.Tensor,  # noqa: ARG002
        output: torch.Tensor,
    ):
        """A forward hook for extracting neuron activations.

        Args:
            module: The module that this hook is registered with.
            input_: Module input.
            output: The module's output.
        """
        # Only do something if recording is not paused and we are below the requested number of steps
        if self._new_trial:
            self.start_new_trial()

        # FIXME: Remove the training regime check.
        # Right now, activations cannot be extracted for SB3-contrib models during training.
        # See https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/issues/299
        if not module.training and (not self.paused) and (self.step < self.steps):
            # Extract the relevant activations for this layer and store it.
            for population, activations in self.monitor.get_activations(output).items():
                self.history[population][-1][self.step] = activations
        self.step += 1

    def start_new_trial(self):
        """Start recording activations for a new trial."""
        shapes = self.monitor.get_population_shapes()

        for population in self.history:
            self.history[population].append(
                np.ma.masked_all((self.steps, *shapes[population]))
            )
        self.step = 0
        self._new_trial = False

    def pause(self):
        """Pause recording."""
        self.paused = True

    def resume(self):
        """Resume recording."""
        self.paused = False
