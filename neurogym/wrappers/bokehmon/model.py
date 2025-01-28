# --------------------------------------
from torch import nn

# --------------------------------------
from bokeh.models import Tabs
from bokeh.models import TabPanel

# --------------------------------------
from neurogym import logger
from neurogym.wrappers.bokehmon.layers.base import LayerBase
from neurogym.wrappers.bokehmon.layers.linear import LinearLayerMonitor
from neurogym.config.components.monitor import NetParam


class ModelMonitor:

    monitor_types = {
        nn.Linear: LinearLayerMonitor,
    }

    def __init__(
        self,
        model: nn.Module,
        name: str = None,
    ):
        '''
        A Torch model monitor.

        Args:
            model (nn.Module):
                The model being monitored.

            name (str, optional):
                The name of this monitor. Defaults to None.
        '''

        # The model that the modules belong to.
        self.model = model

        # The model name.
        self.name = name

        # Model-level callbacks.
        self.callbacks = {}

        # The layers being monitored.
        self.layers: dict[str, LayerBase] = {}

    def _plot(self) -> TabPanel:
        """
        Render the information tracked by this monitor in a tab.

        Returns:
            TabPanel:
                The tab containing sub-components with various types of information.
        """

        return TabPanel(
            title=self.name,
            child=Tabs(
                tabs=[layer._plot() for layer in self.layers.values()],
            ),
        )

    def add_layer(
        self,
        module: nn.Module,
        params: list[NetParam],
        name: str = None,
    ) -> LayerBase:
        """
        Add a module to the list of modules to monitor.

        Args:
            module (nn.Module):
                The module to add.

            params (list[ParameterType]):
                A list of parameter to monitor.

            name (str):
                The name of this module.
        """

        # Get the class of this module
        cls = module.__class__

        # Make sure that we have a name for the module
        if name is None:
            name = f"Layer {len(self.layers)} | {cls.__name__}"

        # Get the monitor class for this parameter type
        monitor_type = self.monitor_types.get(cls, None)
        if monitor_type is None:
            logger.error(
                f"A monitor for layers of type '{cls}' has not been implemented yet."
            )
            return

        # Create a monitor
        self.layers[name] = monitor_type(module, params, name)
        self.layers[name]._start_trial()

        return self.layers[name]

    def _start_trial(self):
        """
        Start monitoring parameters for a new trial.
        """

        for layer in self.layers.values():
            layer._start_trial()
