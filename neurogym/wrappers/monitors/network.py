from typing import ClassVar

from bokeh.models import TabPanel, Tabs
from torch import nn

import neurogym as ngym
from neurogym.wrappers.monitors.layers.base import LayerMonitorBase


class NetworkMonitor:

    def __init__(
        self,
        network: nn.Module,
        phases: set[ngym.MonitorPhase] | None = None,
        name: str | None = None,
    ):
        """A network monitor.

        Args:
            network (nn.Module):
                The network being monitored.

            phases (ngym.MonitorPhase):
                Monitoring phases.

            name (str, optional):
                The name of this monitor. Defaults to None.
        """
        # The network that the modules belong to.
        self.network = network

        # The name of this monitor.
        self.name = name

        # The phases during which the parameters should be monitored.
        if phases is None:
            phases = set()
        self.phases = set(phases)

        # The layers being monitored.
        self.layers: dict[str, LayerMonitorBase] = {}

    def plot(self) -> TabPanel:
        """Render the information tracked by this monitor in a tab.

        Returns:
            TabPanel:
                The tab containing sub-components with various types of information.
        """
        return TabPanel(
            title=self.name,
            child=Tabs(
                tabs=[layer.plot() for layer in self.layers.values()],
            ),
        )

    def register_hooks(self):
        """Register network-level hooks.

        TODO: Populate.
        """
        pass

    def add_layer(
        self,
        layer: nn.Module,
        params: set[ngym.NetParam],
        name: str | None = None,
    ) -> LayerMonitorBase | None:
        """Add a layer to the set of layers to monitor.

        Args:
            layer (nn.Module):
                A layer (Torch nn.Module) to monitor.

            params (list[ngym.NetParam]):
                A list of parameters to monitor for that layer.

            name (str):
                The name of this layer.
        """
        # Get the class type of this layer.
        cls = layer.__class__

        # Make sure that we have a name for the layer
        if name is None:
            name = f"Layer {len(self.layers)} | {cls.__name__}"

        # Get the monitor class for this parameter type.
        monitor_type = LayerMonitorBase.layer_monitors().get(cls)
        if monitor_type is None:
            ngym.logger.error(
                f"A monitor for layers of type '{cls}' has not been implemented yet.",
            )
            return None

        # Create a monitor
        self.layers[name] = monitor_type(layer, params, self.phases, name)
        return self.layers[name]

    def start_trial(self):
        """Start monitoring parameters for a new trial."""
        for layer in self.layers.values():
            layer.start_trial()

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
        for layer in self.layers.values():
            layer.set_phase(phase)
            layer.start_trial()
