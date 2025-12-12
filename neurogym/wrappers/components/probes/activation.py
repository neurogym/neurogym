import numpy as np
import torch
from torch import nn

from neurogym.wrappers.components.modules import LayerProbeBase


class ActivationProbe:
    def __init__(
        self,
        module: nn.Module,
        steps: int | None = None,
        name: str | None = None,
        populations: set[str] | None = None,
    ):
        """Activation monitoring component.

        Args:
            module: The layer being monitored.
            steps: The number of steps to record activations for.
            name: The name of this monitor.
            populations: Neuron populations to monitor.

        """
        # If not set, use the largest representable 64-bit unsigned integer
        self.steps: int = steps or int(np.iinfo(np.uint64).max)
        self.probe = LayerProbeBase.get_probe(module, self._fw_activation_hook, populations)
        self.name = name or module.__class__.__name__

        # The neuron count
        self.paused = False
        self.history: dict[str, list[list[np.ndarray]]] = {population: [] for population in self.probe.populations}
        self.step: int = 0
        self._new_trial: bool = False
        self._monitor_during_training = False
        self.start_new_trial()

    @property
    def init_new_trial(self) -> bool:
        """Get the _new_trial attribute."""
        return self._new_trial

    @init_new_trial.setter
    def init_new_trial(self, value: bool):
        """Set the _new_trial attribute.

        Args:
            value: The new value.
        """
        self._new_trial = value

    @property
    def monitor_during_training(self):
        """A property that determines whether activation trace can be monitored during training.

        NOTE: To be removed when the following issue is resolved:
        See https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/issues/299
        """
        return self._monitor_during_training

    @monitor_during_training.setter
    def monitor_during_training(self, value: bool):
        """Set the _monitor_during_training attribute.

        Args:
            value: The new value.
        """
        self._monitor_during_training = value

    def _fw_activation_hook(
        self,
        module: nn.Module,
        input_: torch.Tensor,  # noqa: ARG002
        output: torch.Tensor,
    ):
        """A forward hook for extracting neuron activations.

        Args:
            module: The module that this hook is registered with.
            input_: Module input.
            output: The module's output.
        """
        # Only do something if recording is not paused and we are below the requested number of steps.
        if self._new_trial:
            self.start_new_trial()

        # FIXME: Remove the training regime check.
        # Right now, activations cannot be extracted for SB3-contrib models during training.
        # See https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/issues/299
        if (
            not self.paused
            and ((not module.training) or (module.training and self._monitor_during_training))
            and (self.step < self.steps)
        ):
            # Extract and store the activations from the requested neuron populations.
            for population, activations in self.probe.get_activations(output).items():
                self.history[population][-1].append(activations)
            self.step += 1

    def start_new_trial(self):
        """Start recording activations for a new trial."""
        for population in self.history:
            self.history[population].append([])
        self.step = 0
        self._new_trial = False

    def pause(self):
        """Pause recording."""
        self.paused = True

    def resume(self):
        """Resume recording."""
        self.paused = False
