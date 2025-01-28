# --------------------------------------
import torch
from torch import nn

# --------------------------------------
from bokeh.models import TabPanel


class ComponentBase:

    def __init__(
        self,
        module: nn.Module,
    ):
        '''
        A base class for all monitor components.

        Args:
            module (nn.Module):
                The module being monitored.
        '''
        self.module = module

    def process(
        self,
        module: nn.Module,
        input_: torch.Tensor,
        output: torch.Tensor,
        trial: int,
        step: int,
    ):
        """
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

        raise NotImplementedError("Please implement this method in a derived class.")

    def _start_trial(self):
        """
        Start monitoring parameters for a new trial.
        """

        pass

    def _plot(self) -> TabPanel:
        """
        Render this component as a tab.

        Raises:
            NotImplementedError:
                Raised when trying to use the base class directly.

        Returns:
            TabPanel:
                The current component as a tab.
        """

        raise NotImplementedError("Please implement this method in a derived class.")
