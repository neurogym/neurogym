from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Wrapper

from neurogym import _SB3_INSTALLED
from neurogym.config.base import LOCAL_DIR
from neurogym.config.config import Config
from neurogym.utils.functions import ensure_dir, iso_timestamp
from neurogym.utils.logging import logger
from neurogym.utils.plotting import visualize_run

if _SB3_INSTALLED:
    from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

if TYPE_CHECKING:
    from collections.abc import Callable

    from neurogym.core import TrialEnv


class Monitor(Wrapper):
    """Monitor class to log, visualize, and evaluate NeuroGym environment behavior.

    Wraps a NeuroGym TrialEnv to track actions, rewards, and performance metrics,
    save them to disk, and optionally generate trial visualizations. Supports logging
    at trial or step level, with configurable frequency and verbosity.

    Args:
        env: The NeuroGym environment to wrap.
        config: Optional configuration source (Config object, TOML file path, or dictionary).
        name: Optional monitor name; defaults to the environment class name.
        trigger: When to save data ("trial" or "step").
        interval: How often to save data, in number of trials or steps.
        plot_create: Whether to generate and save visualizations of environment behavior.
        plot_steps: Number of steps to visualize in each plot.
        ext: Image file extension for saved plots (e.g., "png").
        step_fn: Optional custom step function to override the environment's.
        verbose: Whether to log information when logging or saving data.
        level: Logging verbosity level (e.g., "INFO", "DEBUG").
        log_trigger: When to log progress ("trial" or "step").
        log_interval: How often to log, in trials or steps.

    Attributes:
        config: Final validated configuration object.
        data: Collected behavioral data for each completed trial.
        data_eval: Evaluation data collected during policy evaluation runs.
        cum_reward: Cumulative reward for the current trial.
        num_tr: Number of completed trials.
        t: Step counter (used when trigger is "step").
        save_dir: Directory where data and plots are saved.
    """

    metadata = {  # noqa: RUF012
        "description": (
            "Saves relevant behavioral information: rewards, actions, observations, new trial, ground truth."
        ),
        "paper_link": None,
        "paper_name": None,
    }

    def __init__(
        self,
        env: TrialEnv,
        config: Config | str | Path | None = None,
        name: str | None = None,
        trigger: str = "trial",
        interval: int = 1000,
        verbose: bool = True,
        plot_create: bool = False,
        plot_steps: int = 1000,
        ext: str = "png",
        step_fn: Callable | None = None,
    ) -> None:
        super().__init__(env)
        self.env = env
        self.step_fn = step_fn

        cfg: Config
        if config is None:
            config_dict = {
                "env": {"name": env.unwrapped.__class__.__name__},
                "monitor": {
                    "name": name or "Monitor",
                    "trigger": trigger,
                    "interval": interval,
                    "verbose": verbose,
                    "plot": {
                        "create": plot_create,
                        "step": plot_steps,
                        "title": env.unwrapped.__class__.__name__,
                        "ext": ext,
                    },
                },
                "local_dir": LOCAL_DIR,
            }
            cfg = Config.model_validate(config_dict)
        elif isinstance(config, (str, Path)):
            cfg = Config(config_file=config)
        else:
            cfg = config  # type: ignore[arg-type]

        self.config: Config = cfg

        # Assign names for the environment and/or the monitor if they are empty
        if len(self.config.env.name) == 0:
            self.config.env.name = self.env.unwrapped.__class__.__name__
        if len(self.config.monitor.name) == 0:
            self.config.monitor.name = self.__class__.__name__

        # data to save
        self.data: dict[str, list] = {"action": [], "reward": [], "cum_reward": [], "performance": []}
        # Evaluation-only data collected during the policy run (not saved to disk)
        self.data_eval: dict[str, Any] = {}
        self.cum_reward = 0.0
        if self.config.monitor.trigger == "step":
            self.t = 0
        self.num_tr = 0

        # Directory for saving plots
        save_dir_name = f"{self.config.env.name}/{iso_timestamp()}"
        self.save_dir = ensure_dir(self.config.local_dir / save_dir_name)

        # Figures
        if self.config.monitor.plot.create:
            self.stp_counter = 0
            self.ob_mat: list = []
            self.act_mat: list = []
            self.rew_mat: list = []
            self.gt_mat: list = []
            self.perf_mat: list = []

        # Ensure the reset function is called
        initial_ob, *_ = env.reset()
        self.initial_ob = initial_ob

    def reset(self, seed=None):
        """Reset the environment.

        Args:
            seed: Random seed for the environment

        Returns:
            The initial observation from the environment reset
        """
        self.cum_reward = 0
        return super().reset(seed=seed)

    def step(self, action: Any, collect_data: bool = True) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Execute one environment step.

        This method:
        1. Takes a step in the environment
        2. Collects data if sv_fig is enabled
        3. Saves data when a trial completes and saving conditions are met

        Args:
            action: The action to take in the environment
            collect_data: If True, collect and save data

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.step_fn is not None:
            obs, rew, terminated, truncated, info = self.step_fn(action)
        else:
            obs, rew, terminated, truncated, info = self.env.step(action)
        self.cum_reward += rew
        if self.config.monitor.plot.create:
            self.store_data(obs, action, rew, info)
        if self.config.monitor.trigger == "step":
            self.t += 1
        if info.get("new_trial", False):
            self.num_tr += 1
            self.data["action"].append(action)
            self.data["reward"].append(rew)
            self.data["cum_reward"].append(self.cum_reward)
            self.cum_reward = 0
            for key in info:
                if key not in self.data:
                    self.data[key] = [info[key]]
                else:
                    self.data[key].append(info[key])

            # save data
            save = (
                self.t >= self.config.monitor.interval
                if self.config.monitor.trigger == "step"
                else self.num_tr % self.config.monitor.interval == 0
            )
            if save and collect_data:
                # Create save path with pathlib for cross-platform compatibility
                save_path = self.save_dir / f"trial_{self.num_tr}.npz"
                np.savez(save_path, **self.data)

                if self.config.monitor.verbose:
                    logger.info("--------------------")
                    logger.info(f"Data saved to: {save_path}")
                    logger.info(f"Number of trials: {self.num_tr}")
                    logger.info(f"Average reward: {np.mean(self.data['reward'])}")
                    logger.info(f"Average performance: {np.mean(self.data['performance'])}")
                    logger.info("--------------------")
                self.reset_data()
                if self.config.monitor.plot.create:
                    self.stp_counter = 0
                if self.config.monitor.trigger == "step":
                    self.t = 0
        return obs, rew, terminated, truncated, info

    def reset_data(self) -> None:
        """Reset all data containers to empty lists."""
        for key in self.data:
            self.data[key] = []

    def store_data(self, obs: Any, action: Any, rew: float, info: dict[str, Any]) -> None:
        """Store data for visualization figures.

        Args:
            obs: Current observation
            action: Current action
            rew: Current reward
            info: Info dictionary from environment
        """
        if self.stp_counter <= self.config.monitor.plot.step:
            self.ob_mat.append(obs)
            self.act_mat.append(action)
            self.rew_mat.append(rew)
            if "gt" in info:
                self.gt_mat.append(info["gt"])
            else:
                self.gt_mat.append(-1)
            if "performance" in info:
                self.perf_mat.append(info["performance"])
            else:
                self.perf_mat.append(-1)
            self.stp_counter += 1
        elif len(self.rew_mat) > 0:
            fname = self.save_dir / f"task_{self.num_tr:06d}.{self.config.monitor.plot.ext}"
            obs_mat = np.array(self.ob_mat)
            act_mat = np.array(self.act_mat)
            visualize_run(
                ob=obs_mat,
                actions=act_mat,
                gt=self.gt_mat,
                rewards=self.rew_mat,
                performance=self.perf_mat,
                fname=fname,
                name=self.config.monitor.plot.title,
                env=self.env,
                initial_ob=self.initial_ob,
            )
            self.ob_mat = []
            self.act_mat = []
            self.rew_mat = []
            self.gt_mat = []
            self.perf_mat = []

    def evaluate_policy(
        self,
        num_trials: int = 100,
        model: Any | None = None,
        verbose: bool = True,
    ) -> dict[str, float | list[float]]:
        """Evaluates the average performance of the RL agent in the environment.

        This method runs the given model (or random policy if None) on the
        environment for a specified number of trials and collects performance
        metrics.

        Args:
            num_trials: Number of trials to run for evaluation
            model: The policy model to evaluate (if None, uses random actions)
            verbose: If True, prints progress information
        Returns:
            dict: Dictionary containing performance metrics:
                - mean_performance: Average performance (if reported by environment)
                - mean_reward: Proportion of positive rewards
                - performances: List of performance values for each trial
                - rewards: List of rewards for each trial.
        """
        # Reset environment
        obs, _ = self.env.reset()

        # Initialize hidden states
        states = None
        episode_starts = np.array([True])

        # Tracking variables
        rewards = []
        cum_reward = 0.0
        cum_rewards = []
        performances = []
        self.data_eval = {"action": [], "reward": []}
        # Initialize trial count
        trial_count = 0

        # Run trials
        while trial_count < num_trials:
            if model is not None:
                if _SB3_INSTALLED and isinstance(model.policy, RecurrentActorCriticPolicy):
                    action, states = model.predict(obs, state=states, episode_start=episode_starts, deterministic=True)
                else:
                    action, _ = model.predict(obs, deterministic=True)
            else:
                action = self.env.action_space.sample()

            # Use collect_data=False to avoid saving evaluation data
            obs, reward, _, _, info = self.step(action, collect_data=False)
            # Update episode_starts after each step
            episode_starts = np.array([False])
            cum_reward += reward

            if info.get("new_trial", False):
                trial_count += 1
                rewards.append(reward)
                cum_rewards.append(cum_reward)
                cum_reward = 0.0
                if "performance" in info:
                    performances.append(info["performance"])

                if verbose and trial_count % 1000 == 0:  # FIXME: why is this value hardcoded?
                    logger.info(f"Completed {trial_count}/{num_trials} trials")

                self.data_eval["action"].append(action)
                self.data_eval["reward"].append(reward)
                for key in info:
                    if key not in self.data_eval:
                        self.data_eval[key] = [info[key]]
                    else:
                        self.data_eval[key].append(info[key])

                # Reset states at the end of each trial
                states = None
                episode_starts = np.array([True])

        # Calculate metrics
        performance_array = np.array([p for p in performances if p != -1])
        reward_array = np.array(rewards)
        cum_reward_array = np.array(cum_rewards)

        # Convert lists to numpy arrays for easier downstream analysis
        for key in self.data_eval:
            if key != "trial":
                self.data_eval[key] = np.array(self.data_eval[key])

        return {
            "rewards": rewards,
            "mean_reward": float(np.mean(reward_array > 0)) if len(reward_array) > 0 else 0,
            "cum_rewards": cum_rewards,
            "mean_cum_reward": float(np.mean(cum_reward_array)) if len(cum_reward_array) > 0 else 0,
            "performances": performances,
            "mean_performance": float(np.mean(performance_array)) if len(performance_array) > 0 else 0,
        }

    def plot_training_history(
        self,
        figsize: tuple[int, int] = (12, 6),
        save_fig: bool = True,
        plot_performance: bool = True,
    ) -> plt.Figure | None:
        """Plot rewards and performance training history from saved data files with one data point per trial.

        Args:
            figsize: Figure size as (width, height) tuple
            save_fig: Whether to save the figure to disk
            plot_performance: Whether to plot performance in a separate plot
        Returns:
            matplotlib figure object
        """
        files = sorted(self.save_dir.glob("*.npz"))

        if not files:
            logger.warning("No data files found matching pattern: *.npz")
            return None

        logger.info(f"Found {len(files)} data files")

        # Arrays to hold average values
        avg_rewards_per_file = []
        avg_cum_rewards_per_file = []
        avg_performances_per_file = []
        file_indices = []
        total_trials = 0

        for file in files:
            data = np.load(file, allow_pickle=True)

            if "reward" in data:
                rewards = data["reward"]
                if len(rewards) > 0:
                    avg_rewards_per_file.append(np.mean(rewards))
                    total_trials += len(rewards)
                    file_indices.append(total_trials)

            if "cum_reward" in data:
                cum_rewards = data["cum_reward"]
                if len(cum_rewards) > 0:
                    avg_cum_rewards_per_file.append(np.mean(cum_rewards))

            if "performance" in data:
                perfs = data["performance"]
                if len(perfs) > 0:
                    avg_performances_per_file.append(np.mean(perfs))

        file_indices = [0, *file_indices]
        avg_rewards_per_file = [0, *avg_rewards_per_file]
        avg_cum_rewards_per_file = [0, *avg_cum_rewards_per_file]
        avg_performances_per_file = [0, *avg_performances_per_file]

        fig, axes = plt.subplots(1, 2 if plot_performance else 1, figsize=figsize)
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        # 1. Rewards and Cumulative Rewards plot
        ax1 = axes[0]

        if len(avg_rewards_per_file) == len(file_indices):
            ax1.plot(file_indices, avg_rewards_per_file, "o-", color="blue", label="Avg Reward", linewidth=2)
        if len(avg_cum_rewards_per_file) == len(file_indices):
            ax1.plot(file_indices, avg_cum_rewards_per_file, "s--", color="red", label="Avg Cum Reward", linewidth=2)

        ax1.set_xlabel("Trials")
        ax1.set_ylabel("Reward / Cumulative Reward")
        common_ylim = (-0.05, 1.05)
        ax1.set_ylim(common_ylim)
        ax1.set_title("Reward and Cumulative Reward per File")

        ax1.grid(True, which="both", axis="y", linestyle="--", alpha=0.7)
        ax1.legend(loc="center right", bbox_to_anchor=(1, 0.2))

        # 2. Optional: Performances plot
        if plot_performance and len(axes) > 1:
            ax2 = axes[1]
            if len(avg_performances_per_file) == len(file_indices):
                ax2.plot(file_indices, avg_performances_per_file, "o-", color="green", linewidth=2)
            ax2.set_xlabel("Trials")
            ax2.set_ylabel("Average Performance (0-1)")
            ax2.set_ylim(common_ylim)
            ax2.set_title("Average Performance per File")

            ax2.grid(True, which="both", axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        fig.subplots_adjust(top=0.8)

        for ax in fig.axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.suptitle(
            f"Training History for {self.config.env.name}\n({len(files)} data files, {total_trials} total trials)",
            fontsize=12,
        )

        if save_fig:
            save_path = self.config.local_dir / f"{self.config.env.name}_training_history.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Figure saved to {save_path}")

        return fig
