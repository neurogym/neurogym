"""Plotting functions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Env, make  # using ngym.make would lead to circular import
from matplotlib import animation

from neurogym import _SB3_INSTALLED
from neurogym.config.components.plot import PlotConfig
from neurogym.config.config import config
from neurogym.core import TrialEnv
from neurogym.utils.decorators import suppress_during_pytest
from neurogym.utils.logging import logger

if _SB3_INSTALLED:
    from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
    from stable_baselines3.common.vec_env import DummyVecEnv


@suppress_during_pytest(
    ValueError,
    message="This may be due to a small sample size; please increase to get reasonable results.",
)
def plot_env(
    env: TrialEnv | Env,
    num_steps: int = 200,
    num_trials: int | None = None,
    def_act: int | None = None,
    model=None,
    name: str | None = None,
    legend: bool = True,
    ob_traces: list | None = None,
    fig_kwargs: dict | None = None,
    fname: str | None = None,
    plot_performance: bool = True,
    plot_config: PlotConfig | None = None,
):
    """Plot environment with agent.

    Args:
        env: Already built neurogym task or its name.
        num_steps: Number of steps to run the task for.
        num_trials: If not None, the number of trials to run.
        def_act: If not None (and model=None), the task will be run with the
            specified action.
        model: If not None, the task will be run with the actions predicted by
            model, which so far is assumed to be created and trained with the
            stable-baselines3 toolbox:
            (https://stable-baselines3.readthedocs.io/en/master/)
        name: Title to show on the rewards panel.
        legend: Whether to show the legend for actions panel or not.
        ob_traces: If != [] observations will be plot as traces, with the labels
            specified by ob_traces.
        fig_kwargs: Figure properties admitted by matplotlib.pyplot.subplots() function
        fname: If not None, save fig or movie to fname.
        plot_performance: Whether to show the performance subplot (default: True).
        plot_config: Plot configuration (experimental). If set to None, the global configuration is used.
    """
    # We don't use monitor here because:
    # 1) env could be already prewrapped with monitor
    # 2) monitor will save data and so the function will need a folder
    if fig_kwargs is None:
        fig_kwargs = {}
    if ob_traces is None:
        ob_traces = []
    if isinstance(env, str):
        env = make(env, disable_env_checker=True)
    if name is None:
        name = type(env).__name__
    data = run_env(
        env=env,
        num_steps=num_steps,
        num_trials=num_trials,
        def_act=def_act,
        model=model,
    )
    # Find trial start steps (0-based)
    trial_starts_step_indices = np.where(np.array(data["actions_end_of_trial"]) != -1)[0] + 1
    # Shift again for plotting (since steps are 1-based)
    trial_starts_axis = trial_starts_step_indices + 1

    if plot_config is None:
        plot_config = config.plot

    return visualize_run(
        data["ob"],
        data["actions"],
        gt=data["gt"],
        rewards=data["rewards"],
        legend=legend,
        performance=data["perf"] if plot_performance else None,
        states=data["states"],
        name=name,
        ob_traces=ob_traces,
        fig_kwargs=fig_kwargs,
        env=env,
        initial_ob=data["initial_ob"],
        fname=fname,
        trial_starts=trial_starts_axis,
        plot_config=plot_config,
    )


def run_env(
    env: TrialEnv | Env,
    num_steps: int = 200,
    num_trials: int | None = None,
    def_act: int | None = None,
    model=None,
) -> dict:
    """Run the given environment with the.

    Args:
        env: A NeuroGym environment.
        num_steps: Number of steps to run the task for.
        num_trials: Number of trials to run.
        def_act: Preset action to pass to the environment.
        model: Model (agent) learning from the environment.

    Returns:
        A dictionary containing the results of the trial.
    """
    observations = []
    ob_cum = []
    state_mat = []
    rewards = []
    actions = []
    actions_end_of_trial = []
    gt = []
    perf = []
    if _SB3_INSTALLED and isinstance(env, DummyVecEnv):
        ob = env.reset()
    else:
        ob, _ = env.reset()

    initial_ob = ob.copy()

    ob_cum_temp = ob.copy()

    # Initialize hidden states
    states = None
    episode_starts = np.array([True])

    if num_trials is not None:
        num_steps = int(1e5)  # Overwrite num_steps value

    trial_count = 0
    for _ in range(int(num_steps)):
        if model is not None:
            if _SB3_INSTALLED and isinstance(model.policy, RecurrentActorCriticPolicy):
                action, states = model.predict(ob, state=states, episode_start=episode_starts, deterministic=True)
            else:
                action, _ = model.predict(ob, deterministic=True)
            if isinstance(action, float | int):
                action = [action]
            if (states is not None) and (len(states) > 0):
                state_mat.append(states)
        elif def_act is not None:
            action = def_act
        else:
            action = env.action_space.sample()
        if _SB3_INSTALLED and isinstance(env, DummyVecEnv):
            ob, rew, terminated, info = env.step(action)
        else:
            ob, rew, terminated, _truncated, info = env.step(action)
        # Update episode_starts after each step
        episode_starts = np.array([False])
        ob_cum_temp += ob
        ob_cum.append(ob_cum_temp.copy())
        if isinstance(info, list):
            info = info[0]
            ob_aux = ob[0]
            # TODO: Fix these and remove the ignore directives
            rew = rew[0]  # type: ignore[index]
            terminated = terminated[0]  # type: ignore[index]
            action = action[0]
        else:
            ob_aux = ob

        if terminated:
            env.reset()
        observations.append(ob_aux)
        rewards.append(rew)
        actions.append(action)
        if "gt" in info:
            gt.append(info["gt"])
        else:
            gt.append(0)

        if info["new_trial"]:
            actions_end_of_trial.append(action)
            perf.append(info["performance"])
            ob_cum_temp = np.zeros_like(ob_cum_temp)
            trial_count += 1
            # Reset states at the end of each trial
            states = None
            episode_starts = np.array([True])
            if num_trials is not None and trial_count >= num_trials:
                break
        else:
            actions_end_of_trial.append(-1)
            perf.append(-1)

    if model is not None and len(state_mat) > 0:  # noqa: SIM108
        # states = np.array(state_mat)  # noqa: ERA001
        # states = states[:, 0, :]  # noqa: ERA001
        states = None  # TODO: Fix this
    else:
        states = None

    return {
        "ob": np.array(observations).astype(float),
        "ob_cum": np.array(ob_cum).astype(float),
        "rewards": rewards,
        "actions": actions,
        "perf": perf,
        "actions_end_of_trial": actions_end_of_trial,
        "gt": gt,
        "states": states,
        "initial_ob": initial_ob,
    }


@suppress_during_pytest(
    ValueError,
    message="This may be due to a small sample size; please increase to get reasonable results.",
)
def visualize_run(
    ob: np.ndarray,
    actions: np.ndarray,
    gt: np.ndarray | None = None,
    rewards: np.ndarray | None = None,
    performance: np.ndarray | None = None,
    states: np.ndarray | None = None,
    legend: bool = True,
    ob_traces: list | None = None,
    name: str = "",
    fname: str | None = None,
    fig_kwargs: dict | None = None,
    env: TrialEnv | Env | None = None,
    initial_ob: np.ndarray | None = None,
    trial_starts: list | None = None,
    plot_config: PlotConfig | None = None,
) -> None:
    """Visualize a run in a simple environment.

    Args:
        ob: NumPy array of observation (n_step, n_unit).
        actions: NumPy array of action (n_step, n_unit)
        gt: NumPy array of groud truth.
        rewards: NumPy array of rewards.
        performance: NumPy array of performance (if set to `None` performance plotting will be skipped).
        states: NumPy array of network states
        name: Title to show on the rewards panel and name to save figure.
        fname: Optional name for the file where the figure should be saved.
        legend: Whether to show the legend for actions panel.
        ob_traces: If a non-empty listis provided, observations will be plot as traces,
            with the labels specified by ob_traces
        fig_kwargs: Figure properties admitted by matplotlib.pyplot.subplots() function
        env: Environment class for extra information
        initial_ob: Initial observation to be used to align with actions
        trial_starts: List of trial start indices, 1-based
        plot_config: Optional plot configuration.
    """
    if fig_kwargs is None:
        fig_kwargs = {}
    ob = np.array(ob)
    actions = np.array(actions)

    if initial_ob is None:
        initial_ob = ob[0].copy()

    # Align observation with actions by inserting an initial obs from env
    ob = np.insert(ob, 0, initial_ob, axis=0)
    # Trim last obs to match actions
    ob = ob[:-1]

    if plot_config is None:
        plot_config = config.plot

    if len(ob.shape) == 2:
        return plot_env_1dbox(  # type: ignore[no-any-return]
            ob,
            actions,
            gt=gt,
            rewards=rewards,
            performance=performance,
            states=states,
            legend=legend,
            ob_traces=ob_traces,
            name=name,
            fname=fname,
            fig_kwargs=fig_kwargs,
            env=env,
            trial_starts=trial_starts,
            plot_config=plot_config,
        )
    if len(ob.shape) == 4:
        return plot_env_3dbox(ob, fname=fname, env=env)  # type: ignore[no-any-return]

    msg = f"{ob.shape=} not supported."
    raise ValueError(msg)


def _style_axis(
    ax: plt.Axes,
    xticks: np.ndarray | None = None,
    xtick_labels: list[str] | None = None,
    yticks: np.ndarray | None = None,
    ytick_labels: list[str] | None = None,
    plot_config: PlotConfig | None = None,
):
    """Set standard grid styling for plot background.

    Args:
        ax: Matplotlib plot axis.
        xticks: x-axis tick positions.
        xtick_labels: x-axis tick labels.
        yticks: y-axis tick positions.
        ytick_labels: y-axis tick labels.
        plot_config: Optional plot configuration.
    """
    line_width = 1.0

    if plot_config is None:
        plot_config = config.plot

    ax.set_facecolor("whitesmoke")
    ax.grid(True, color="white", linestyle="-", linewidth=line_width)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if xticks is not None:
        ax.set_xticks(
            xticks,
            labels=xtick_labels or [str(xt) for xt in xticks],
            fontsize=plot_config.font.xtick.get_size(),
        )

    if yticks is not None:
        ax.set_yticks(
            yticks,
            labels=ytick_labels or [str(yt) for yt in yticks],
            fontsize=plot_config.font.ytick.get_size(),
        )


def _plot_observations(
    ax: plt.Axes,
    steps: np.ndarray,
    ob: np.ndarray,
    traces: list | None = None,
    env: TrialEnv | Env | None = None,
    trial_starts: np.ndarray | None = None,
    xticks: np.ndarray | None = None,
    xtick_labels: list[str] | None = None,
    ylabel: str = "Observation",
    title: str = "",
    name: str = "",
    legend: bool = True,
    legend_props: dict | None = None,
    plot_config: PlotConfig | None = None,
) -> None:
    """Plot an array of observed values.

    Args:
        ax: A Matplotlib plot axis.
        steps: An array containing the step numbers to plot against.
        ob: An array of observed values.
        traces: An array of observation traces.
        env: A NeuroGym environment.
        trial_starts: Trial start times.
        xticks: x-axis tick values.
        xtick_labels: x-axis tick labels.
        ylabel: y-axis tick labels.
        title: Axis title.
        name: Name of the environment.
        legend: Toggle for including a legend.
        legend_props: Legend options (if a legend is included).
        plot_config: Plot configuration.
    """
    line_width = 1.0

    if plot_config is None:
        plot_config = config.plot

    if traces:
        if len(traces) != ob.shape[1]:
            msg = f"Please provide label for each of the {ob.shape[1]} traces in the observations."
            raise ValueError(msg)

        # Plot all traces first
        for ind_tr, tr in enumerate(traces):
            ax.plot(steps, ob[:, ind_tr], label=tr, lw=line_width)

        # Compute ticks and labels
        yticks = []
        ytick_labels = []

        # Find fixation index (if exists)
        fix_idx = next((i for i, tr in enumerate(traces) if "fix" in tr.lower()), None)

        if fix_idx is not None:
            yticks.append(np.max(ob[:, fix_idx]))
            ytick_labels.append("Fix. cue")

            # All other indices are stimuli
            stim_means = [np.mean(ob[:, i]) for i in range(len(traces)) if i != fix_idx]
            if stim_means:
                yticks.append(np.mean(stim_means))
                ytick_labels.append("Stimuli")
        else:
            # No fixation, all are stimuli
            yticks.append(np.mean([np.mean(ob[:, i]) for i in range(len(traces))]))
            ytick_labels.append("Stimuli")

        if legend:
            if legend_props is None:
                legend_props = {}
            ax.legend(**legend_props)
        if trial_starts is not None:
            for t_start in trial_starts:
                ax.axvline(t_start, linestyle="--", color="grey", alpha=0.7)
        ax.set_xlim((0.5, len(steps) + 1))
        ax.set_yticks(yticks, labels=ytick_labels, fontsize=plot_config.font.ytick.get_size())
    else:
        ax.imshow(ob.T, aspect="auto", origin="lower")
        if env and hasattr(env.observation_space, "name"):
            # Plot environment annotation
            yticks = []
            ytick_labels = []
            for key, val in env.observation_space.name.items():
                yticks.append((np.min(val) + np.max(val)) / 2)
                ytick_labels.append(key)
            ax.set_yticks(yticks, labels=ytick_labels, fontsize=plot_config.font.ytick.get_size())
        else:
            ax.set_yticks([])

    if name:
        ax.set_title(title or f"{name} env", fontproperties=plot_config.font.title)
    ax.set_ylabel(ylabel, fontproperties=plot_config.font.label)

    # Style the axis
    _style_axis(ax, xticks, xtick_labels, plot_config=plot_config)


def _plot_actions(
    ax: plt.Axes,
    steps: np.ndarray,
    actions: np.ndarray,
    gt: np.ndarray | None = None,
    env: TrialEnv | Env | None = None,
    trial_starts: np.ndarray | None = None,
    xticks: np.ndarray | None = None,
    xtick_labels: list[str] | None = None,
    gt_colors: str = "gkmcry",
    legend: bool = True,
    legend_props: dict | None = None,
    plot_config: PlotConfig | None = None,
) -> None:
    line_width = 1.0

    if plot_config is None:
        plot_config = config.plot

    if len(actions.shape) > 1:
        # Changes not implemented yet
        ax.plot(steps, actions, marker="+", label="Actions", lw=line_width)
    else:
        ax.plot(steps, actions, marker="+", label="Actions", lw=line_width)
    if gt is not None:
        gt = np.array(gt)
        if len(gt.shape) > 1:
            for ind_gt in range(gt.shape[1]):
                ax.plot(
                    steps,
                    gt[:, ind_gt],
                    f"--{gt_colors[ind_gt]}",
                    label=f"Ground truth {ind_gt}",
                    lw=line_width,
                )
        else:
            ax.plot(steps, gt, f"--{gt_colors[0]}", label="Ground truth", lw=line_width)
    if trial_starts is not None:
        for t_start in trial_starts:
            ax.axvline(t_start, linestyle="--", color="grey", alpha=0.7)
    ax.set_xlim((0.5, len(steps) + 1))
    ax.set_ylabel("Action", fontproperties=plot_config.font.label)

    if legend:
        if legend_props is None:
            legend_props = {}
        ax.legend(**legend_props)

    if env and hasattr(env.action_space, "name"):
        yticks = []
        ytick_labels = []
        for key, val in env.action_space.name.items():
            if isinstance(val, list | tuple | np.ndarray | range):
                for v in val:
                    yticks.append(v)
                    ytick_labels.append(f"{str(key).capitalize()}_{v}")
            else:  # single int
                yticks.append(val)
                ytick_labels.append(str(key).capitalize())
        ax.set_yticks(yticks, labels=ytick_labels, fontsize=plot_config.font.ytick.get_size())

    # Style the axis
    _style_axis(ax, xticks, xtick_labels, plot_config=plot_config)


def _plot_rewards(
    ax: plt.Axes,
    steps: np.ndarray,
    rewards: np.ndarray,
    env: TrialEnv | Env | None = None,
    trial_starts: np.ndarray | None = None,
    xticks: np.ndarray | None = None,
    xtick_labels: list[str] | None = None,
    legend: bool = True,
    legend_props: dict | None = None,
    plot_config: PlotConfig | None = None,
) -> None:
    line_width = 1.0

    if plot_config is None:
        plot_config = config.plot

    rewards = np.array(rewards)

    # Plot rewards if provided
    ax.plot(steps, rewards, "r", label="Rewards", lw=line_width)
    ax.set_ylabel("Reward", fontproperties=plot_config.font.label)

    if legend:
        if legend_props is None:
            legend_props = {}
        ax.legend(**legend_props)
    if trial_starts is not None:
        for t_start in trial_starts:
            ax.axvline(t_start, linestyle="--", color="grey", alpha=0.7)
    ax.set_xlim((0.5, len(steps) + 1))

    yticks = []
    ytick_labels = []
    if env and hasattr(env, "rewards") and env.rewards is not None:
        if isinstance(env.rewards, dict):
            for key, val in env.rewards.items():
                yticks.append(val)
                ytick_labels.append(f"{str(key)[:5].title()} {val:0.2f}")
        else:
            for val in env.rewards:
                yticks.append(val)
                ytick_labels.append(f"{val:0.2f}")
    else:
        yticks = [rewards.min(), (rewards.min() + rewards.max()) / 2, rewards.max()]

    _style_axis(ax, xticks, xtick_labels, np.array(yticks), ytick_labels, plot_config=plot_config)


def _plot_performance(
    ax: plt.Axes,
    steps: np.ndarray,
    performance: np.ndarray,
    trial_starts: np.ndarray | None = None,
    xticks: np.ndarray | None = None,
    xtick_labels: list[str] | None = None,
    legend: bool = True,
    legend_props: dict | None = None,
    plot_config: PlotConfig | None = None,
) -> None:
    line_width = 1.0

    if plot_config is None:
        plot_config = config.plot

    ax.plot(steps, performance, "m", label="Performance", lw=line_width)
    ax.set_ylabel("Performance", fontproperties=plot_config.font.label)
    performance = np.array(performance)
    mean_perf = np.mean(performance[performance != -1])
    ax.set_title(
        f"Mean performance: {np.round(mean_perf, 2)}",
        fontproperties=plot_config.font.title,
    )
    if legend:
        if legend_props is None:
            legend_props = {}
        ax.legend(**legend_props)
    if trial_starts is not None:
        for t_start in trial_starts:
            ax.axvline(t_start, linestyle="--", color="grey", alpha=0.7)
    ax.set_xlim((0.5, len(steps) + 1))

    yticks = np.array([performance.min(), (performance.min() + performance.max()) / 2, performance.max()])

    _style_axis(ax, xticks, xtick_labels, yticks, plot_config=plot_config)


def _plot_states(
    ax: plt.Axes,
    states: np.ndarray,
    performance: np.ndarray | None = None,
    rewards: np.ndarray | None = None,
    xticks: np.ndarray | None = None,
    xtick_labels: list[str] | None = None,
    plot_config: PlotConfig | None = None,
) -> None:
    if plot_config is None:
        plot_config = config.plot
    plt.imshow(states[:, int(states.shape[1] / 2) :].T, aspect="auto")

    if performance is not None or rewards is not None:
        if xticks is not None:
            if xtick_labels is None:
                xtick_labels = xticks.tolist()
            # Show step numbers on x-axis
            ax.set_xticks(xticks, labels=xtick_labels, fontsize=7)
        ax.tick_params(axis="both", labelsize=7)

    ax.set_title("Activity", fontproperties=plot_config.font.title)
    ax.set_ylabel("Neurons", fontproperties=plot_config.font.label)


@suppress_during_pytest(
    ValueError,
    message="This may be due to a small sample size; please increase to get reasonable results.",
)
def plot_env_1dbox(
    ob: np.ndarray,
    actions: np.ndarray,
    gt: np.ndarray | None = None,
    rewards: np.ndarray | None = None,
    performance: np.ndarray | None = None,
    states: np.ndarray | None = None,
    legend: bool = True,
    ob_traces: list | None = None,
    name: str = "",
    fname: str | None = None,
    fig_kwargs: dict | None = None,
    env: TrialEnv | Env | None = None,
    trial_starts: np.ndarray | None = None,
    plot_config: PlotConfig | None = None,
) -> plt.Figure | None:
    """Plot environment with 1-D Box observation space.

    Args:
        ob: Array of observed values.
        actions: Array of actions.
        gt: Array of ground truth values.
        rewards: Array of reward values.
        performance: Array of performance values.
        states: Array of state values.
        legend: Legend toggle.
        ob_traces: List of observation traces.
        name: Name of the environment.
        fname: Name of the file to save the plot to.
        fig_kwargs: Figure configuration.
        env: A NeuroGym environment.
        trial_starts: List of trial start times.
        plot_config: Plot configuration.

    Raises:
        ValueError: Raised if the array of observed values is not 2D.

    Returns:
        A Matplotlib figure.
    """
    if plot_config is None:
        plot_config = config.plot

    if len(ob.shape) != 2:
        msg = "ob must be 2-dimensional."
        raise ValueError(msg)
    steps = np.arange(1, ob.shape[0] + 1)

    n_row = 2  # observation and action
    for extra in [rewards, performance, states]:
        n_row += int(extra is not None)

    gt_colors = "gkmcry"
    if not fig_kwargs:
        fig_kwargs = {
            "sharex": True,
            "figsize": (12, n_row * 1.4),
        }
    f, axes = plt.subplots(n_row, 1, **fig_kwargs)
    i_ax = 0

    legend_props = {
        "loc": "upper right",
        "bbox_to_anchor": (1.01, 1.0, 0.15, 0),
        "mode": "expand",
        "fontsize": 8,
    }

    xticks = np.arange(0, len(steps) + 1, 10)
    xtick_labels = [f"{x}" for x in xticks]

    # Observations
    ax = axes[i_ax]
    i_ax += 1
    _plot_observations(
        ax,
        steps,
        ob,
        ob_traces,
        env=env,
        trial_starts=trial_starts,
        xticks=xticks,
        xtick_labels=xtick_labels,
        title="Observation traces",
        name=name,
        legend=legend,
        legend_props=legend_props,
        plot_config=plot_config,
    )

    # Plot actions
    ax = axes[i_ax]
    i_ax += 1
    _plot_actions(
        ax,
        steps,
        actions,
        gt,
        env=env,
        trial_starts=trial_starts,
        xticks=xticks,
        xtick_labels=xtick_labels,
        gt_colors=gt_colors,
        legend=legend,
        legend_props=legend_props,
        plot_config=plot_config,
    )

    if rewards is not None:
        ax = axes[i_ax]
        i_ax += 1
        _plot_rewards(
            ax,
            steps,
            rewards,
            env=env,
            trial_starts=trial_starts,
            xticks=xticks,
            xtick_labels=xtick_labels,
            legend=legend,
            legend_props=legend_props,
            plot_config=plot_config,
        )

    # Plot performance if provided
    if performance is not None:
        ax = axes[i_ax]
        i_ax += 1
        _plot_performance(
            ax,
            steps,
            performance,
            trial_starts=trial_starts,
            xticks=xticks,
            xtick_labels=xtick_labels,
            legend=legend,
            legend_props=legend_props,
            plot_config=plot_config,
        )

    # Plot states if provided
    if states is not None:
        ax = axes[i_ax]
        i_ax += 1
        _plot_states(
            ax,
            states,
            performance,
            rewards,
            xticks=xticks,
            xtick_labels=xtick_labels,
            plot_config=plot_config,
        )

    ax.set_xlabel("Steps", fontproperties=plot_config.font.label)
    f.align_ylabels(axes)
    f.tight_layout()

    if fname:
        fname = str(fname)
        if not (fname.endswith((".png", ".svg"))):
            fname += ".png"
        f.savefig(fname, dpi=300)
        plt.close(f)

    return f  # type: ignore[no-any-return]


@suppress_during_pytest(
    ValueError,
    message="This may be due to a small sample size; please increase to get reasonable results.",
)
def plot_env_3dbox(
    ob: np.ndarray,
    fname: str = "",
    env: TrialEnv | Env | None = None,
) -> None:
    """Plot environment with 3-D Box observation space.

    Args:
        ob: Array of observation values.
        fname: File name to save the figure to.
        env: A NeuroGym environment.
    """
    ob = ob.astype(np.uint8)  # TODO: Temporary
    fig = plt.figure()
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    ax.axis("off")
    im = ax.imshow(ob[0], animated=True)

    def animate(i, *args, **kwargs):
        im.set_array(ob[i])
        return (im,)

    interval = env.dt if isinstance(env, TrialEnv) else 50
    ani = animation.FuncAnimation(fig, animate, frames=ob.shape[0], interval=interval)
    if fname:
        writer = animation.writers["ffmpeg"](fps=int(1000 / interval))
        fname = str(fname)
        if not fname.endswith(".mp4"):
            fname += ".mp4"
        ani.save(fname, writer=writer, dpi=300)


def plot_rew_across_training(
    folder: str,
    window: int = 500,
    ax: plt.Axes | None = None,
    fkwargs: dict | None = None,
    ytitle: str = "",
    legend: bool = False,
    zline: bool = False,
    metric_name: str = "reward",
) -> None:
    if fkwargs is None:
        fkwargs = {"c": "tab:blue"}
    data = put_together_files(folder)

    line_width = 1.0
    if data:
        sv_fig = False
        if ax is None:
            sv_fig = True
            f, ax = plt.subplots(figsize=(8, 8))
        metric = data[metric_name]
        if isinstance(window, float) and window < 1.0:
            window = int(metric.size * window)
        mean_metric = np.convolve(metric, np.ones((window,)) / window, mode="valid")
        ax.plot(mean_metric, lw=line_width, **fkwargs)  # add color, label etc.
        ax.set_xlabel("trials")
        if not ytitle:
            ax.set_ylabel(f"mean {metric_name} (running window of {window} trials)")
        else:
            ax.set_ylabel(ytitle)
        if legend:
            ax.legend()
        if zline:
            ax.axhline(0, c="k", ls=":")
        if sv_fig:
            f.savefig(
                folder + "/mean_" + metric_name + "_across_training.png",
            )  # FIXME: use pathlib to specify location
    else:
        logger.info(f"No data in: {folder}")


def put_together_files(folder: str) -> dict:
    files = [str(f) for f in Path(folder).glob("/*_bhvr_data*npz")]
    data = {}
    if len(files) > 0:
        files = order_by_suffix(files)
        file_data = np.load(files[0], allow_pickle=True)
        for key in file_data:
            data[key] = file_data[key]

        for ind_f in range(1, len(files)):
            file_data = np.load(files[ind_f], allow_pickle=True)
            for key in file_data:
                data[key] = np.concatenate((data[key], file_data[key]))
        np.savez(folder + "/bhvr_data_all.npz", **data)  # FIXME: use pathlib to specify location
    return data


def order_by_suffix(file_list: list[str]) -> list:
    sfx = [int(x[x.rfind("_") + 1 : x.rfind(".")]) for x in file_list]  # FIXME: use pathlib method to find extension
    return [x for _, x in sorted(zip(sfx, file_list, strict=True))]
