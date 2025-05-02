from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Wrapper

import neurogym as ngym
from neurogym.config.base import LOCAL_DIR
from neurogym.utils.plotting import fig_


class Monitor(Wrapper):
    """Monitor wrapper for NeuroGym environments.

    This class wraps a NeuroGym environment to track trial-based behavioral data,
    optionally log training progress, and generate visualizations. Data is collected
    at the end of each trial and can be saved periodically.

    The monitor can be configured using a `Config` object, a TOML file path, or a dictionary.
    If no config is provided, the constructor parameters act as a fallback to manually configure
    logging, plotting, and environment naming.

    Data is saved based on independent triggers and intervals for logging and plotting.

    Args:
        env: The NeuroGym environment to wrap.
        config: Optional configuration source. It can be:
            - a `Config` object (from `neurogym.Config`)
            - a string or `Path` pointing to a TOML config file
            - a dictionary to be validated as a Config model
        name: Optional name used for monitoring (defaults to env class name). Used only if `config` is not provided.
        plot_trigger: When to generate plots: "trial" or "step". Used only if `config` is not provided.
        plot_value: Frequency of plotting (every N trials or steps). Used only if `config` is not provided.
        log_trigger: When to log training info: "trial" or "step". Used only if `config` is not provided.
        log_value: Frequency of logging (every N trials or steps). Used only if `config` is not provided.
        create: Whether to generate visualizations. Used only if `config` is not provided.
        verbose: If True, prints info when saving data/logging. Used only if `config` is not provided.
        level: Logging level as string, e.g. "INFO", "DEBUG". Used only if `config` is not provided.
        ext: File extension for plot outputs (e.g., "png"). Used only if `config` is not provided.
        step_fn: Optional callable to override the environment's step function.

    Attributes:
        config: The final configuration object used (validated `Config`).
        data: Dictionary storing collected behavioral data (e.g., rewards, actions, performance).
        num_tr: Number of completed trials.
        t: Number of timesteps completed (used if `plot_trigger` is "step").
        save_dir: Path where data and plots are saved.
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
        env: ngym.TrialEnv,
        config: ngym.Config | str | Path | None = None,
        name: str | None = None,
        plot_trigger: str = "trial",
        plot_value: int = 1000,
        log_trigger: str = "trial",
        log_value: int = 1000,
        create: bool = False,
        verbose: bool = True,
        level: str = "INFO",
        ext: str = "png",
        step_fn: Callable | None = None,
    ) -> None:
        super().__init__(env)
        self.env = env
        self.step_fn = step_fn

        log_format = "<magenta>Neurogym</magenta> | <cyan>{time:YYYY-MM-DD@HH:mm:ss}</cyan> | <level>{message}</level>"

        cfg: ngym.Config
        if config is None:
            config_dict = {
                "env": {"name": env.unwrapped.__class__.__name__},
                "monitor": {
                    "name": name or "Monitor",
                    "plot": {
                        "create": create,
                        "ext": ext,
                        "trigger": plot_trigger,
                        "value": plot_value,
                        "title": env.unwrapped.__class__.__name__,
                    },
                    "log": {
                        "verbose": verbose,
                        "format": log_format,
                        "level": level,
                        "trigger": log_trigger,
                        "value": log_value,
                    },
                },
                "local_dir": LOCAL_DIR,
            }
            cfg = ngym.Config.model_validate(config_dict)
        elif isinstance(config, (str, Path)):
            cfg = ngym.Config(config_file=config)
        else:
            cfg = config  # type: ignore[arg-type]

        self.config: ngym.Config = cfg

        # Assign names for the environment and/or the monitor if they are empty
        if len(self.config.env.name) == 0:
            self.config.env.name = self.env.unwrapped.__class__.__name__
        if len(self.config.monitor.name) == 0:
            self.config.monitor.name = self.__class__.__name__

        self._configure_logger()

        self.data: dict[str, list] = {"action": [], "reward": [], "performance": []}
        if self.config.monitor.plot.trigger == "step":
            self.t = 0
        self.num_tr = 0

        # Directory for saving plots
        save_dir_name = f"{self.config.env.name}/{ngym.utils.iso_timestamp()}"
        self.save_dir = ngym.utils.ensure_dir(self.config.local_dir / save_dir_name)

        # Figures
        if self.config.monitor.plot.create:
            self.stp_counter = 0
            self.ob_mat: list = []
            self.act_mat: list = []
            self.rew_mat: list = []
            self.gt_mat: list = []
            self.perf_mat: list = []

    def _configure_logger(self):
        ngym.logger.remove()
        ngym.logger.configure(**self.config.monitor.log.make_config())
        ngym.logger.info("Logger configured.")

    def reset(self, seed=None):
        """Reset the environment.

        Args:
            seed: Random seed for the environment

        Returns:
            The initial observation from the environment reset
        """
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
        if self.config.monitor.plot.create:
            self.store_data(obs, action, rew, info)
        if self.config.monitor.plot.trigger == "step":
            self.t += 1
        if info.get("new_trial", False):
            self.num_tr += 1
            self.data["action"].append(action)
            self.data["reward"].append(rew)
            for key in info:
                if key not in self.data:
                    self.data[key] = [info[key]]
                else:
                    self.data[key].append(info[key])

            # save data
            save = (
                self.t >= self.config.monitor.plot.interval
                if self.config.monitor.plot.trigger == "step"
                else self.num_tr % self.config.monitor.plot.interval == 0
            )
            if save and collect_data:
                # Create save path with pathlib for cross-platform compatibility
                save_path = self.config.local_dir / f"trial_{self.num_tr}.npz"
                np.savez(save_path, **self.data)

                if self.config.monitor.log.verbose:
                    print("--------------------")
                    print(f"Data saved to: {save_path}")
                    print(f"Number of trials: {self.num_tr}")
                    print(f"Average reward: {np.mean(self.data['reward'])}")
                    print(f"Average performance: {np.mean(self.data['performance'])}")
                    print("--------------------")
                self.reset_data()
                if self.config.monitor.plot.create:
                    self.stp_counter = 0
                if self.config.monitor.plot.trigger == "step":
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
        if self.stp_counter <= self.config.monitor.plot.interval:
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
            fig_(
                ob=obs_mat,
                actions=act_mat,
                gt=self.gt_mat,
                rewards=self.rew_mat,
                performance=self.perf_mat,
                fname=fname,
                name=self.config.monitor.plot.title,
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
        performances = []
        rewards = []
        trial_count = 0

        # Run trials
        while trial_count < num_trials:
            if model is not None:
                action, states = model.predict(obs, state=states, episode_start=episode_starts, deterministic=True)
            else:
                action = self.env.action_space.sample()

            # Use collect_data=False to avoid saving evaluation data
            obs, reward, _, _, info = self.step(action, collect_data=False)
            # Update episode_starts after each step
            episode_starts = np.array([False])

            if info.get("new_trial", False):
                trial_count += 1
                rewards.append(reward)
                if "performance" in info:
                    performances.append(info["performance"])

                if verbose and trial_count % 1000 == 0:
                    print(f"Completed {trial_count}/{num_trials} trials")

                # Reset states at the end of each trial
                states = None
                episode_starts = np.array([True])

        # Calculate metrics
        performance_array = np.array([p for p in performances if p != -1])
        reward_array = np.array(rewards)

        return {
            "mean_performance": (float(np.mean(performance_array)) if len(performance_array) > 0 else 0),
            "mean_reward": (float(np.mean(reward_array > 0)) if len(reward_array) > 0 else 0),
            "performances": performances,
            "rewards": rewards,
        }

    def plot_training_history(
        self,
        figsize: tuple[int, int] = (12, 6),
        save_fig: bool = True,
    ) -> plt.Figure | None:
        """Plot rewards and performance training history from saved data files with one data point per trial.

        Args:
            figsize: Figure size as (width, height) tuple
            save_fig: Whether to save the figure to disk
        Returns:
            matplotlib figure object
        """
        files = sorted(self.config.local_dir.glob("*.npz"))

        if not files:
            print("No data files found matching pattern: *.npz")
            return None

        print(f"Found {len(files)} data files")

        # Arrays to hold average values for each file
        avg_rewards_per_file = []
        avg_performances_per_file = []
        file_indices = []  # To store file numbers or trial counts
        total_trials = 0

        # Process each file
        for file in files:
            data = np.load(file, allow_pickle=True)

            # Process rewards
            if "reward" in data:
                rewards = data["reward"]
                if len(rewards) > 0:
                    avg_rewards_per_file.append(np.mean(rewards))
                    total_trials += len(rewards)
                    file_indices.append(total_trials)  # Use cumulative trial count as x-axis value

            # Process performances
            if "performance" in data:
                perfs = data["performance"]
                if len(perfs) > 0:
                    avg_performances_per_file.append(np.mean(perfs))

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 1. Rewards plot
        ax1.plot(file_indices, avg_rewards_per_file, "o-", color="blue", linewidth=2)
        ax1.set_title("Average Reward per File")
        ax1.set_xlabel("Cumulative Trials")
        ax1.set_ylabel("Average Reward")
        ax1.set_ylim(-0.05, 1.05)

        overall_avg_reward = np.mean(avg_rewards_per_file)
        ax1.text(
            0.05,
            0.95,
            f"Overall Avg Reward: {overall_avg_reward:.4f}",
            transform=ax1.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        # 2. Performances plot
        ax2.plot(file_indices, avg_performances_per_file, "o-", color="green", linewidth=2)
        ax2.set_title("Average Performance per File")
        ax2.set_xlabel("Cumulative Trials")
        ax2.set_ylabel("Average Performance (0-1)")
        ax2.set_ylim(-0.05, 1.05)
        overall_avg_perf = np.mean(avg_performances_per_file)
        ax2.text(
            0.05,
            0.95,
            f"Overall Avg Performance: {overall_avg_perf:.4f}",
            transform=ax2.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        plt.tight_layout()
        plt.suptitle(
            f"Training History for {self.config.env.name}\n({len(files)} data files, {total_trials} total trials)",
            fontsize=14,
            y=1.05,
        )

        # Save the figure
        if save_fig:
            save_path = self.config.local_dir / f"{self.config.env.name}_training_history.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to {save_path}")

        return fig
