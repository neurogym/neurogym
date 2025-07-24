import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

from neurogym.envs.registration import make
from neurogym.utils.logging import logger
from neurogym.wrappers.noise import Noise
from neurogym.wrappers.pass_action import PassAction
from neurogym.wrappers.pass_reward import PassReward
from neurogym.wrappers.reaction_time import ReactionTime

# TODO: No tests exist for wrappers from neurogym/wrappers/block.py
# i.e. RandomGroundTruth, ScheduleAttr, ScheduleEnvs, TrialHistoryV2


def test_passaction():
    """Test pass-action wrapper."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")

        env_name = "PerceptualDecisionMaking-v0"
        num_steps = 100

        env = make(env_name)
        env = PassAction(env)
        env.reset()
        for _ in range(num_steps):
            action = env.action_space.sample()
            obs, _rew, terminated, _truncated, _info = env.step(action)
            assert obs[-1] == action, "Previous action is not part of observation"
            if terminated:
                env.reset()


def test_passreward():
    """Test pass-reward wrapper."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")

        env_name = "PerceptualDecisionMaking-v0"
        num_steps = 100

        env = make(env_name)
        env = PassReward(env)
        obs, _ = env.reset()
        for _ in range(num_steps):
            action = env.action_space.sample()
            obs, rew, terminated, _truncated, _info = env.step(action)
            assert obs[-1] == rew, "Previous reward is not part of observation"
            if terminated:
                env.reset()


@pytest.mark.skip(reason="This test is failing, needs more investigation")
def test_reactiontime(
    env_name="PerceptualDecisionMaking-v0",
    num_steps=10000,
    urgency=-0.1,
    thresholds=None,
    verbose=False,
):
    """Test reaction-time wrapper.

    The reaction-time wrapper allows converting a fix duration task into a reaction
    time task. It also allows adding a fix (negative) quantity (urgency) to force
    the network to respond quickly.
    Parameters
    ----------
    env_name : str, optional
        environment to wrap. The default is 'PerceptualDecisionMaking-v0'.
    num_steps : int, optional
        number of steps to run the environment (1000)
    urgency : float, optional
        float value added to the reward (-0.1)
    verbose : boolean, optional
        whether to log observation and reward (False)
    thresholds : list, optional
        list containing the thresholds to make a decision ([-.5, .5])
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")

        if thresholds is None:
            thresholds = [-0.5, 0.5]
        env_args = {"timing": {"fixation": 100, "stimulus": 2000, "decision": 200}}
        env = make(env_name, **env_args)
        env = ReactionTime(env, urgency=urgency)
        env.reset()
        if verbose:
            observations = []
            obs_cum_mat = []
            actions = []
            new_trials = []
            reward = []
        obs_cum = 0
        end_of_trial = False
        for _ in range(num_steps):
            if obs_cum > thresholds[1]:
                action = 1
            elif obs_cum < thresholds[0]:
                action = 2
            else:
                action = 0
            end_of_trial = action != 0
            obs, rew, _terminated, _truncated, info = env.step(action)
            if info["new_trial"]:
                step = 0
                obs_cum = 0
                end_of_trial = False
            else:
                step += 1
                assert not end_of_trial, "Trial still on after making a decision"
                obs_cum += obs[1] - obs[2]
            if verbose:
                observations.append(obs)
                actions.append(action)
                obs_cum_mat.append(obs_cum)
                new_trials.append(info["new_trial"])
                reward.append(rew)
        if verbose:
            observations = np.array(observations)
            _, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
            ax = ax.flatten()
            ax[0].imshow(observations.T, aspect="auto")
            ax[1].plot(actions, label="Actions")
            ax[1].plot(new_trials, "--", label="New trial")
            ax[1].set_xlim([-0.5, len(actions) - 0.5])
            ax[1].legend()
            ax[2].plot(obs_cum_mat, label="cum. observation")
            ax[2].plot([0, len(obs_cum_mat)], [thresholds[1], thresholds[1]], "--", label="upper th")
            ax[2].plot([0, len(obs_cum_mat)], [thresholds[0], thresholds[0]], "--", label="lower th")
            ax[2].set_xlim([-0.5, len(actions) - 0.5])
            ax[3].plot(reward, label="reward")
            ax[3].set_xlim([-0.5, len(actions) - 0.5])


@pytest.mark.skip(reason="This test is failing, needs more investigation")
def test_noise(
    env="PerceptualDecisionMaking-v0",
    margin=0.01,
    perf_th=0.7,
    num_steps=100000,
    verbose=False,
):
    """Test noise wrapper.

    The noise wrapper allows adding noise to the full observation received by the
    network. It also offers the option of fixxing a specific target performance
    that the wrapper will assure by modulating the magnitude of the noise added.
    Parameters
    ----------
    env_name : str, optional
        environment to wrap. The default is 'PerceptualDecisionMaking-v0'.
    num_steps : int, optional
        number of steps to run the environment (1000)
    verbose : boolean, optional
        whether to log observation and reward (False)
    margin : float, optional
        margin allowed when comparing actual and expected performances (0.01)
    perf_th : float, optional
        target performance for the noise wrapper (0.7)
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")

        env_args = {"timing": {"fixation": 100, "stimulus": 200, "decision": 200}}
        env = make(env, **env_args)
        env = Noise(env, perf_th=perf_th)
        env.reset()
        perf = []
        std_mat = []
        std_noise = 0
        for _ in range(num_steps):
            rng = np.random.default_rng()
            action = env.action_space.sample() if rng.random() < std_noise else env.gt_now
            _obs, _rew, terminated, _truncated, info = env.step(action)
            if "std_noise" in info:
                std_noise = info["std_noise"]
            if verbose and info["new_trial"]:
                perf.append(info["performance"])
                std_mat.append(std_noise)
            if terminated:
                env.reset()
        actual_perf = np.mean(perf[-5000:])
        if verbose:
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot([0, len(perf)], [perf_th, perf_th], "--")
            plt.plot(np.convolve(perf, np.ones((100,)) / 100, mode="valid"))
            plt.subplot(2, 1, 2)
            plt.plot(std_mat)
        assert np.abs(actual_perf - perf_th) < margin, f"Actual performance: {actual_perf}, expected: {perf_th}"
