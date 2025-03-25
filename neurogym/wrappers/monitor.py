from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

import bokeh.resources
import numpy as np
import panel as pn
from bokeh.models import Column, Paragraph, Row, Tabs  # type: ignore[attr-defined]

from gymnasium import Wrapper
from torch import nn

import bokeh

import neurogym as ngym
from neurogym.utils.plotting import fig_
from neurogym.wrappers.monitors.network import NetworkMonitor

class Monitor(Wrapper):
    metadata: dict[str, str | None] = {  # noqa: RUF012
        "description": (
            "Saves relevant behavioral information: rewards, actions, observations, new trial, ground truth."
        ),
        "paper_link": None,
        "paper_name": None,
    }

    @dataclass
    class Trial:
        actions: list[np.ndarray] = field(default_factory=list)
        rewards: list[np.ndarray] = field(default_factory=list)
        observations: list[np.ndarray] = field(default_factory=list)
        info: dict = field(default_factory=lambda: defaultdict(list))

    def __init__(
        self,
        env: ngym.TrialEnv,
        step_fn: Callable | None = None,
        name: str | None = None,
    ):
        """A monitoring wrapper driven by Bokeh.

        Saves relevant behavioral information:
            - Rewards
            - Actions
            - Observations
            - New trials
            - Ground truth
            - Policy (model) parameters and activations

        Args:
            env (TrialEnv):
                The environment to work with on.

            step_fn (Callable, optional):
                An optional custom step function. Defaults to None.

            name (str, optional):
                The name of the monitor. Defaults to None.
        """
        super().__init__(env)

        # Environment and step function
        # ==================================================
        self.env = env
        if step_fn is None:
            step_fn = env.step
        self.step_fn = step_fn

        # Monitor name
        self.name = name

        # Global variable monitoring, such as rewards
        # and observations
        # ==================================================
        self.trial = Monitor.Trial()
        self.trials: list[Monitor.Trial] = []

        # Paths and directory names for saving data
        # ==================================================
        # Root directory for saving data
        self.save_dir = ngym.utils.mkdir(ngym.conf.monitor.save_dir)

        # Location for saving data
        self.data_dir = ngym.utils.mkdir(self.save_dir / "data")

        # Location for saving plots
        self.plot_dir = ngym.utils.mkdir(self.save_dir / "plots")

        # Callbacks
        self.callbacks: dict[str, Callable] = {}

        # Visualisation
        # ==================================================
        self.bm = None
        self.server: pn.Server = None
        self.max_t = 0
        self.total_steps = 0

        # Models being monitored.
        #
        # Each model will appear as a tab with sub-tabs
        # representing layers, with further sub-tabs
        # for activations, weights and other parameters.
        # ==================================================
        self.networks: dict[str, NetworkMonitor] = {}

    @property
    def cur_step(self) -> int:
        """The current step for the current the trial.

        Note that this is not the same as t, which represents actual time
        (# of time steps times the time delta).

        Returns:
            int:
                The current time step.
        """
        return int(self.env.unwrapped.t)  # type: ignore[attr-defined]

    @property
    def cur_timestep(self) -> int:
        """The current time as determined by the time delta (dt).

        Returns:
            int:
                The current time.
        """
        return int(self.env.unwrapped.t * self.env.dt)  # type: ignore[attr-defined]

    @property
    def cur_trial(self) -> int:
        """The number of trials that have been executed so far.

        Returns:
            int:
                The number of trials.
        """
        return int(self.env.unwrapped.num_tr)  # type: ignore[attr-defined]

    @property
    def trigger(self) -> int:
        """Log / save trigger.

        A property that determines when the monitor should
        save the data that is currently in the trial buffer.

        Returns:
            int:
                The trigger variable.
        """
        return int(
            (
                self.cur_timestep
                if ngym.conf.monitor.trigger == "timestep"
                else self.cur_trial
            ),
        )

    def add_network(
        self,
        network: nn.Module,
        phases: set[ngym.MonitorPhase] | None = None,
        name: str | None = None,
    ) -> NetworkMonitor:
        if name is None:
            name = f"Network {len(self.networks)}"

        self.networks[name] = NetworkMonitor(network, phases, name)
        return self.networks[name]

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        return self.env.reset()

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool, bool, dict]:
        """This method makes the agent take a single action in the environment.

        Args:
            action (np.ndarray):
                The action that the agent is taking.

        Returns:
            tuple[np.ndarray, np.ndarray, bool, bool, dict]:
                A tuple containing:
                    1. A new observation of the environment.
                    2. The reward from the environment.
                    3. A flag indicating if the trial was terminated.
                    4. A flag indicating if the trial was truncated.
                    5. Extra information about the environment.
        """
        # Take an action in the environment and
        # retrieve the reward and a new observation.
        observation, reward, terminated, truncated, info = self.step_fn(action)

        # Store the action, reward, observation and
        # other info from the environment
        self.trial.actions.append(action)
        self.trial.rewards.append(reward)
        self.trial.observations.append(observation)
        for k, v in info.items():
            self.trial.info[k].append(v)

        # Extract the trigger variable
        trigger = self.trigger

        # Create a plot
        if (
            ngym.conf.monitor.plot.save
            and trigger > 0
            and trigger % ngym.conf.monitor.plot.interval == 0
        ):
            self._save_plot()

        self.total_steps += 1

        # Update the trial information
        if info["new_trial"]:
            # HACK
            # Gymnasium expects a certain format for the info dictionary,
            # and (at last) the `terminated` variable to be set.
            # ==================================================
            info["episode"] = {
                "r": sum(self.trial.rewards),
                "l": self.cur_timestep,
            }
            terminated = True

            # ==================================================
            if ngym.conf.log.verbose and trigger % ngym.conf.log.interval == 0:
                # Display some progress info
                self._update_log()

                # Save the current trial data
                self.store_data()

            # Start a new trial
            self.start_new_trial()

        return observation, reward, terminated, truncated, info

    def start_new_trial(self):
        """Reset all buffers and variables associated with the current trial."""
        for model in self.networks.values():
            model.start_new_trial()

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
        for network in self.networks.values():
            network.set_phase(phase)

    def store_data(self):
        """Stores data about the current trial into a NumPy archive."""
        # Store the trial and start a new one
        self.trials.append(self.trial)
        self.trial = Monitor.Trial()

        data = {
            "actions": self.trial.actions,
            "rewards": self.trial.rewards,
            "observations": self.trial.observations,
            "info": self.trial.info,
            "steps": self.cur_timestep,
        }

        np.savez(self.data_dir / f"trial-{self.cur_trial}-info.npz", **data)

    def _update_log(self):
        """Print some useful information about the progress of learning or evaluation."""
        # Log some output
        # ==================================================
        avg_reward = sum(self.trial.rewards) if len(self.trial.rewards) > 0 else 0
        log_name = f"{self.name} | " if self.name else ""

        self.max_t = max(self.cur_timestep, self.max_t)

        total_steps = self.total_steps * ngym.conf.monitor.dt

        msg = " | ".join(
            [
                f"{log_name}Trial: {self.cur_trial:>5d}",
                f"Time: {self.cur_timestep:>5d} / (max {self.max_t:>5d}, total {total_steps:>5d})",
                f"Avg. reward: {avg_reward:>2.3f}",
            ],
        )

        ngym.logger.info(msg)

    def _save_plot(self):
        """Produce and save a plot at a certain step."""
        actions = np.array(self.trial.actions)
        rewards = np.array(self.trial.rewards)
        observations = np.array(self.trial.observations)
        ground_truth = self.trial.info.get("gt", np.full_like(rewards, -1))
        # FIXME: This is not working at the moment because the performance
        # is assigned only once at the end of a trial.
        performance = np.concatenate(  # noqa: F841
            [
                trial.info.get("performance", np.full_like(rewards, -1))
                for trial in self.trials
            ],
        )

        fname = (
            self.plot_dir / f"task_{self.cur_trial:06d}.{ngym.conf.monitor.plot.ext}"
        )

        fig_(
            ob=observations,
            actions=actions,
            gt=ground_truth,
            rewards=rewards,
            fname=fname,
        )

    def _get_model(self):

        return pn.Row(
            pn.Column(
                pn.pane.Str(
                    self.name,
                    styles={"font-size": "3em"},
                ),
                pn.Tabs(
                    *[(name, mon.plot()) for name, mon in self.networks.items()],
                    tabs_location="left",
                ),
            )
        )

    def plot_browser(self):
        """Display the plot in a browser tab."""

        self._get_model().show()

        # pn.extension()
        # if self.server is not None:
        #     self.server.stop()
        #     self.server = None
        # self.server = pn.serve(self._get_model().servable(), threaded=True)

    def plot_notebook(self):
        """Display the plot inside a notebook."""

        if not ngym.utils.is_notebook():
            ngym.logger.error("Not running inside a notebook.")
            return

        pn.extension()
        return self._get_model()

    def get_traces(
        self,
        nets: str | list[str] | None = None,
        layers: str | list[str] | None = None,
        params: ngym.NetParam | list[ngym.NetParam] | None = None,
        phases: ngym.MonitorPhase | list[ngym.MonitorPhase] | None = None,
    ):

        # Make sure that if the arguments are single elements,
        # they are converted to lists
        nets = [nets] if isinstance(nets, str) else nets
        layers = [layers] if isinstance(layers, str) else layers
        params = [params] if isinstance(params, ngym.NetParam) else params
        phases = [phases] if isinstance(phases, ngym.MonitorPhase) else phases

        traces = {}
        # Extract the monitors for the requested networks
        # ==================================================
        if nets is None:
            nets = list(self.networks.keys())
        net_monitors = {k: self.networks[k] for k in nets if k in self.networks}

        for net_name, net_mon in net_monitors.items():
            net_traces = traces.setdefault(net_name, {})

            # Extract the monitors for the requested layers
            # ==================================================
            if layers is None:
                layers = list(net_mon.layer_monitors.keys())

            layer_monitors = {
                k: net_mon.layer_monitors[k]
                for k in layers
                if k in net_mon.layer_monitors
            }

            for layer_name, layer_mon in layer_monitors.items():
                layer_traces = net_traces.setdefault(layer_name, {})

                # Extract the monitors for the requested parameters
                # ==================================================
                if params is None:
                    params = list(layer_mon.param_monitors.keys())

                param_monitors = {
                    k: layer_mon.param_monitors[k]
                    for k in params
                    if k in layer_mon.param_monitors
                }

                for param_name, param_mon in param_monitors.items():
                    param_traces = layer_traces.setdefault(param_name, {})

                    # Extract the monitors for the requested phases
                    # ==================================================
                    if phases is None:
                        phases = list(param_mon.history.keys())

                    for k in phases:
                        if k in param_mon.history:
                            param_traces[k] = param_mon.history[k]

        return traces
