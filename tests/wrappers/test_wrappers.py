import warnings

import numpy as np
import pytest

from neurogym.envs.registration import make
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


@pytest.mark.skip(reason="Trials not ending when they should.")
def test_reactiontime():
    """Test reaction-time wrapper.

    The reaction-time wrapper allows converting a fix duration task into a reaction
    time task. It also allows adding a fix (negative) quantity (urgency) to force
    the network to respond quickly.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")

        env_name = "PerceptualDecisionMaking-v0"
        env_args = {"timing": {"fixation": 100, "stimulus": 2000, "decision": 200}}
        urgency = -0.1
        thresholds = [-0.5, 0.5]
        num_steps = 200

        env = make(env_name, **env_args)
        env = ReactionTime(env, urgency=urgency)
        env.reset()
        obs_cum = 0
        step = 0
        end_of_trial = False

        for i in range(num_steps):
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
                assert not end_of_trial, f"Trial did not end after decision ({action}) on iteration {i}, step {step}."
                obs_cum += obs[1] - obs[2]


def test_noise():
    """Test noise wrapper.

    The noise wrapper allows adding noise to the full observation received by the network.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")

        env_name = "PerceptualDecisionMaking-v0"
        env_args = {"timing": {"fixation": 100, "stimulus": 200, "decision": 200}}
        num_steps = 200

        env = make(env_name, **env_args)
        env = Noise(env)
        env.reset()
        perf = []
        std_noise = 0.2

        for _ in range(num_steps):
            rng = np.random.default_rng()
            action = env.action_space.sample() if rng.random() < std_noise else env.gt_now
            _obs, _rew, terminated, _truncated, info = env.step(action)
            if "std_noise" in info:
                std_noise = info["std_noise"]
            if info["new_trial"]:
                perf.append(info["performance"])
            if terminated:
                env.reset()

        mean_performance = np.mean(perf[-50:])
        assert mean_performance < 1
