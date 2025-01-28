# --------------------------------------
from typing import Callable

# --------------------------------------
import numpy as np

# --------------------------------------
from torch import nn

# --------------------------------------
from gymnasium import Wrapper

# --------------------------------------
from collections import defaultdict

# --------------------------------------
import panel as pn

# --------------------------------------
from dataclasses import dataclass
from dataclasses import field

# --------------------------------------
from bokeh.models import Tabs
from bokeh.models import Column
from bokeh.models import Paragraph

# --------------------------------------
from neurogym import conf
from neurogym import logger
from neurogym import utils
from neurogym import TrialEnv
from neurogym.utils.plotting import fig_
from neurogym.wrappers.bokehmon.model import ModelMonitor


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
        env: TrialEnv,
        step_fn: Callable = None,
        name: str = None,
    ):
        """
        A monitoring wrapper driven by Bokeh.

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

        # Global variable monitoring, such as rewards
        # and observations
        # ==================================================
        self.trial = Monitor.Trial()
        self.trials = []

        # Paths and directory names for saving data
        # ==================================================
        # Root directory for saving data
        self.save_dir = utils.mkdir(conf.monitor.save_dir)

        # Location for saving data
        self.data_dir = utils.mkdir(self.save_dir / "data")

        # Location for saving plots
        self.plot_dir = utils.mkdir(self.save_dir / "plots")

        # Callbacks
        self.callbacks = {}

        # Monitor name
        self.name = name

        # Visualisation
        # ==================================================
        self.bm = None
        self.server = None
        self.max_t = 0
        self.total_steps = 0

        # Models being monitored.
        #
        # Each model will appear as a tab with sub-tabs
        # representing layers, with further sub-tabs
        # for activations, weights and other parameters.
        # ==================================================
        self.models: dict[str, ModelMonitor] = {}

    @property
    def cur_step(self) -> int:
        """
        The current step for the current the trial.
        Note that this is not the same as t, which represents actual time
        (# of time steps times the time delta).

        Returns:
            int:
                The current time step.
        """
        return int(self.env.unwrapped.t)

    @property
    def cur_timestep(self) -> int:
        """
        The current time as determined by the time delta (dt).

        Returns:
            int:
                The current time.
        """
        return int(self.env.unwrapped.t * self.env.dt)

    @property
    def cur_trial(self) -> int:
        """
        The number of trials that have been executed so far.

        Returns:
            int:
                The number of trials.
        """
        return int(self.env.unwrapped.num_tr)

    @property
    def trigger(self) -> int:
        """
        A property that determines when the monitor should
        save the data that is currently in the trial buffer.

        Returns:
            int:
                The trigger variable.
        """
        return int(
            self.cur_timestep if conf.monitor.trigger == "timestep" else self.cur_trial
        )

    def add_model(
        self,
        model: nn.Module,
        name: str = None,
    ) -> ModelMonitor:

        if name is None:
            name = f"Model {len(self.models)}"

        self.models[name] = ModelMonitor(model, name)
        return self.models[name]

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        return self.env.reset()

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool, bool, dict]:
        """
        This method makes the agent take a single action in the environment.

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
            conf.monitor.plot.save
            and trigger > 0
            and trigger % conf.monitor.plot.interval == 0
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

            if conf.log.verbose and trigger % conf.log.interval == 0:
                # Display some progress info
                self._update_log()

                # Save the current trial data
                self._store_data()

            # Start a new trial
            self._start_new_trial()

        return observation, reward, terminated, truncated, info

    def _start_new_trial(self):
        """
        Reset all buffers and variables associated with the current trial.
        """

        for model in self.models.values():
            model._start_trial()

    def _store_data(self):
        """
        Stores data about the current trial into a NumPy archive.
        """

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
        """
        Print some useful information about the progress of learning or evaluation.
        """

        # Log some output
        # ==================================================
        avg_reward = sum(self.trial.rewards) if len(self.trial.rewards) > 0 else 0
        log_name = f"{self.name} | " if self.name else ""

        if self.cur_timestep > self.max_t:
            self.max_t = self.cur_timestep
        logger.info(
            f"{log_name}Trial: {self.cur_trial:>5d} | Time: {self.cur_timestep:>5d}  / (max {self.max_t:>5d}, total {self.total_steps * conf.monitor.dt:>5d}) | Avg. reward: {avg_reward:>2.3f}"
        )

    def _save_plot(self):
        """
        Produce and save a plot at a certain step.
        """

        actions = np.array(self.trial.actions)
        rewards = np.array(self.trial.rewards)
        observations = np.array(self.trial.observations)
        ground_truth = self.trial.info.get("gt", np.full_like(rewards, -1))
        performance = np.concatenate(
            [
                trial.info.get("performance", np.full_like(rewards, -1))
                for trial in self.trials
            ]
        )

        fname = self.plot_dir / f"task_{self.cur_trial:06d}.{conf.monitor.plot.ext}"

        f = fig_(
            ob=observations,
            actions=actions,
            gt=ground_truth,
            rewards=rewards,
            # performance=performance,
            fname=fname,
        )

    def plot(self):
        """
        Display the Bokehmon app in a browser tab.
        """

        pn.extension()

        if self.bm is None:
            self.bm = Column(
                Paragraph(text=self.name, styles={"font-size": "3em"}),
                Tabs(
                    tabs=[model._plot() for model in self.models.values()],
                    tabs_location="left",
                ),
            )
            self.server = pn.serve(self.bm)

        else:
            self.server.show("/")
