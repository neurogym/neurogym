from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Wrapper

from neurogym.utils.plotting import fig_


class Monitor(Wrapper):
    """Monitor wrapper for NeuroGym environments.

    This class wraps NeuroGym environments to monitor and collect behavioral data
    during training and evaluation. It saves relevant behavioral information such as
    rewards, actions, observations, new trial flags, and ground truth.

    Data collection behavior:
    - The Monitor records data points ONLY at the end of trials (when info["new_trial"]=True).
    - Each data point represents one complete behavioral trial, not individual timesteps.
    - For NeuroGym tasks, this typically occurs when the agent makes a decision in the
      decision period, or when the trial is aborted.
    - Data is saved to disk at regular intervals (e.g., every 1000 trials) and internal
      data containers are reset after each save.

    Args:
        env: The NeuroGym environment to wrap
        folder: Path to folder where data will be saved. If None, uses a temporary directory.
        sv_per: How often to save data (in trials or timesteps, depending on sv_stp)
        sv_stp: Unit for sv_per, either "trial" or "timestep"
        verbose: If True, prints information about average reward and number of trials
        sv_fig: If True, saves figures visualizing the experiment structure
        num_stps_sv_fig: Number of trial steps to include in each visualization figure
        name: Optional name suffix for saved files
        fig_type: File format for saved figures (e.g., 'png', 'svg', 'pdf')
        step_fn: Optional custom step function to use instead of env.step

    Attributes:
        data: Dictionary containing behavioral data (actions, rewards, etc.)
        num_tr: Number of trials completed
        t: Number of timesteps completed (when sv_stp="timestep")
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
        env: Any,
        folder: str | None = None,
        sv_per: int = 100000,
        sv_stp: str = "trial",
        verbose: bool = False,
        sv_fig: bool = False,
        num_stps_sv_fig: int = 100,
        name: str = "",
        fig_type: str = "png",
        step_fn: Callable | None = None,
    ) -> None:
        super().__init__(env)
        self.env = env
        self.num_tr: int = 0
        self.step_fn = step_fn
        # data to save
        self.data: dict[str, list] = {"action": [], "reward": []}
        self.sv_per = sv_per
        self.sv_stp = sv_stp
        self.fig_type = fig_type
        if self.sv_stp == "timestep":
            self.t: int = 0
        self.verbose = verbose
        self.folder: str = "tmp" if folder is None else folder
        Path(self.folder).mkdir(parents=True, exist_ok=True)

        # seeding
        self.sv_name: str = str(Path(self.folder) / f"{self.env.__class__.__name__}_bhvr_data_{name}_")
        # figure
        self.sv_fig = sv_fig
        self.name: str = name
        if self.sv_fig:
            self.num_stps_sv_fig = num_stps_sv_fig
            self.stp_counter: int = 0
            self.ob_mat: list = []
            self.act_mat: list = []
            self.rew_mat: list = []
            self.gt_mat: list = []
            self.perf_mat: list = []

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
        if self.step_fn:
            obs, rew, terminated, truncated, info = self.step_fn(action)
        else:
            obs, rew, terminated, truncated, info = self.env.step(action)
        if self.sv_fig:
            self.store_data(obs, action, rew, info)
        if self.sv_stp == "timestep":
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
            save = False
            save = self.t >= self.sv_per if self.sv_stp == "timestep" else self.num_tr % self.sv_per == 0
            if save and collect_data:
                # Create save path with pathlib for cross-platform compatibility
                save_path = f"{self.sv_name}{self.num_tr}.npz"
                np.savez(save_path, **self.data)

                if self.verbose:
                    print("--------------------")
                    print(f"Data saved to: {save_path}")
                    print(f"Number of trials: {self.num_tr}")
                    print(f"Average reward: {np.mean(self.data['reward'])}")
                    print(f"Average performance: {np.mean(self.data['performance'])}")
                    print("--------------------")
                self.reset_data()
                if self.sv_fig:
                    self.stp_counter = 0
                if self.sv_stp == "timestep":
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
        if self.stp_counter <= self.num_stps_sv_fig:
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
            fname = (
                Path(self.folder) / f"{self.env.unwrapped.__class__.__name__}_task_{self.num_tr:06d}.{self.fig_type}"
            )
            obs_mat = np.array(self.ob_mat)
            act_mat = np.array(self.act_mat)
            fig_(
                ob=obs_mat,
                actions=act_mat,
                gt=self.gt_mat,
                rewards=self.rew_mat,
                performance=self.perf_mat,
                fname=fname,
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
            "mean_performance": float(np.mean(performance_array)) if len(performance_array) > 0 else 0,
            "mean_reward": float(np.mean(reward_array > 0)) if len(reward_array) > 0 else 0,
            "performances": performances,
            "rewards": rewards,
        }

    def plot_training_history(
        self,
        window_size: int | None = None,
        figsize: tuple[int, int] = (12, 6),
        save_fig: bool = True,
    ) -> plt.Figure | None:
        """Plot rewards and performance history from saved data files with one data point per trial.

        Args:
            window_size: Size of the moving average window (default: auto-calculate based on data size)
            figsize: Figure size as (width, height) tuple
            save_fig: Whether to save the figure to disk

        Returns:
            matplotlib figure object
        """
        env_name = self.env.unwrapped.__class__.__name__
        log_folder = self.folder

        base_path = Path(log_folder)
        files = sorted(base_path.glob(f"{env_name}_bhvr_data_{self.name}_*.npz"))

        if not files:
            print(f"No data files found matching pattern: {env_name}_bhvr_data_{self.name}_*.npz")
            return None

        print(f"Found {len(files)} data files")

        all_rewards: list[float] = []
        all_performances: list[float] = []
        current_trial = 0

        for file in files:
            data = np.load(file, allow_pickle=True)

            if "reward" in data:
                rewards = data["reward"]
                all_rewards.extend(rewards)
                current_trial += len(rewards)

            if "performance" in data:
                perfs = data["performance"]
                all_performances.extend(perfs)

        if window_size is None:
            window_size = max(1, min(100, len(all_rewards) // 20))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 1. Rewards plot
        if len(all_rewards) > 0:
            trial_indices = np.arange(1, len(all_rewards) + 1)
            ax1.plot(trial_indices, all_rewards, alpha=0.4, color="blue", label="Raw")

            # Calculate moving average for smoothing
            if len(all_rewards) > window_size:
                weights = np.ones(window_size) / window_size
                smoothed_rewards = np.convolve(all_rewards, weights, mode="valid")
                # Adjust x-axis for convolution
                smoothed_indices = trial_indices[window_size - 1 :]

                ax1.plot(
                    smoothed_indices,
                    smoothed_rewards,
                    linewidth=2,
                    color="blue",
                    label=f"Moving Avg (w={window_size})",
                )

            ax1.set_title("Reward per Trial")
            ax1.set_xlabel("Trial Number")
            ax1.set_ylabel("Reward")
            ax1.legend(loc="upper right")

            avg_reward = np.mean(all_rewards)
            ax1.text(
                0.05,
                0.95,
                f"Overall Avg: {avg_reward:.4f}",
                transform=ax1.transAxes,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )
        else:
            ax1.text(
                0.5,
                0.5,
                "Insufficient reward data",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax1.transAxes,
            )

        # 2. Performance plot
        if len(all_performances) > 0:
            trial_indices = np.arange(1, len(all_performances) + 1)
            valid_mask = np.array(all_performances) != -1
            valid_indices = trial_indices[valid_mask]
            valid_performances = np.array(all_performances)[valid_mask]

            if len(valid_performances) > 0:
                ax2.plot(valid_indices, valid_performances, alpha=0.4, color="green", label="Raw")

                # Calculate moving average for smoothing
                if len(valid_performances) > window_size:
                    weights = np.ones(window_size) / window_size
                    smoothed_perf = np.convolve(valid_performances, weights, mode="valid")
                    # Adjust x-axis for convolution
                    smoothed_indices = valid_indices[window_size - 1 :]
                    if len(smoothed_indices) > len(smoothed_perf):
                        smoothed_indices = smoothed_indices[: len(smoothed_perf)]

                    ax2.plot(
                        smoothed_indices,
                        smoothed_perf,
                        linewidth=2,
                        color="green",
                        label=f"Moving Avg (w={window_size})",
                    )

                ax2.set_title("Performance per Trial")
                ax2.set_xlabel("Trial Number")
                ax2.set_ylabel("Performance (0-1)")
                ax2.set_ylim(-0.05, 1.05)  # Performance is typically 0-1
                ax2.legend(loc="upper right")

                # Add text with overall average performance
                avg_perf = np.mean(valid_performances)
                ax2.text(
                    0.05,
                    0.95,
                    f"Overall Avg: {avg_perf:.4f}",
                    transform=ax2.transAxes,
                    verticalalignment="top",
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
                )
            else:
                ax2.text(
                    0.5,
                    0.5,
                    "No valid performance data (all -1)",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax2.transAxes,
                )
        else:
            ax2.text(
                0.5,
                0.5,
                "No performance data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax2.transAxes,
            )

        plt.tight_layout()
        plt.suptitle(f"Training History for {env_name}\n({len(all_rewards)} total trials)", fontsize=14, y=1.05)

        # Save the figure
        if save_fig:
            save_path = Path(log_folder) / f"{env_name}_training_history.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to {save_path}")

        return fig
