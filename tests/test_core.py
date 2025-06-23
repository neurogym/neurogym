import warnings

import numpy as np
import pytest

import neurogym as ngym
from neurogym.core import TrialEnv, env_string
from neurogym.utils import spaces


def test_one_step_mismatch():
    """Test the agent gets rewarded if it fixates after seeing fixation cue."""

    class TestEnv(TrialEnv):
        def __init__(self, dt=100) -> None:
            super().__init__(dt=dt)
            self.timing = {"fixation": dt, "go": dt}
            name = {"fixation": 0, "go": 1}
            self.observation_space = spaces.Box(
                -np.inf,
                np.inf,
                shape=(2,),
                dtype=np.float32,
                name=name,
            )
            self.action_space = spaces.Discrete(2)

        def _new_trial(self, **kwargs):
            self.add_period(["fixation", "go"])
            self.add_ob(1, period="fixation", where="fixation")
            self.add_ob(1, period="go", where="go")
            return {}

        def _step(self, action):
            info = {"new_trial": False}
            reward = (action == 0) * 1.0 if self.in_period("fixation") else (action == 1) * 1.0
            terminated = False
            truncated = False
            return self.ob_now, reward, terminated, truncated, info

    env = TestEnv()
    ob, _ = env.reset(options={"no_step": True})
    for i in range(10):
        action = np.argmax(ob)  # fixate if observes fixation input, vice versa
        ob, rew, _terminated, _truncated, _info = env.step(action=action)
        if i > 0:
            assert rew == 1


def test_addob_instep():
    """Test if we can add observation during step."""

    class TestEnv(TrialEnv):
        def __init__(self, dt=100) -> None:
            super().__init__(dt=dt)
            self.timing = {"go": 500}
            self.observation_space = spaces.Box(
                -np.inf,
                np.inf,
                shape=(1,),
                dtype=np.float32,
            )
            self.action_space = spaces.Discrete(3)

        def _new_trial(self, **kwargs):
            trial = {}
            self.add_period("go")
            self.add_ob(1)
            return trial

        def _step(self, action):  # noqa: ARG002
            new_trial = False
            terminated = False
            truncated = False
            reward = 0
            self.add_ob(1)
            return self.ob_now, reward, terminated, truncated, {"new_trial": new_trial}

    env = TestEnv()
    env.reset(options={"no_step": True})
    for i in range(10):
        ob, _rew, _terminated, _truncated, _info = env.step(action=0)
        assert ob[0] == ((i + 1) % 5) + 1  # each trial is 5 steps


class DummyEnv(TrialEnv):
    def __init__(self, dt: int, timing: dict, seed: int = 42) -> None:
        super().__init__(dt=dt)
        self.seed(seed)
        self.timing = timing
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(1,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)

    def _new_trial(self, **kwargs):
        # Add periods
        if self.timing:
            self.add_period(list(self.timing.keys()))
            self.add_ob(1)

        return {}

    def _step(self, action):  # noqa: ARG002
        return self.ob_now, 0, False, False, {"new_trial": False}


@pytest.mark.parametrize(
    ("timing", "expected_stats"),
    [
        # Fixed timing (exact values)
        (
            {"fixation": 300, "stimulus": 500, "decision": 200},
            {"mean": 10, "std": 0, "percentile_95": 10, "max": 10},
        ),
        (
            {},
            {"mean": 0, "std": 0, "percentile_95": 0, "max": 0},
        ),
        # Randomized timing (specific expected values, after fixing the random seed)
        (
            {"fixation": 300, "stimulus": ["uniform", [400, 600]], "decision": 200},
            {"mean": 10, "std": 0, "percentile_95": 10, "max": 10},
        ),
        (
            {"fixation": 300, "stimulus": ["truncated_exponential", [400, 100, 500]], "decision": 200},
            {"mean": 8, "std": 1, "percentile_95": 8, "max": 8},
        ),
        (
            {"fixation": 300, "stimulus": ["choice", [300, 400, 500]], "decision": 200},
            {"mean": 9, "std": 0, "percentile_95": 9, "max": 9},
        ),
    ],
)
def test_trial_length_stats(timing, expected_stats):
    """Test trial length stats for both fixed and randomized timing configurations."""
    dt = 100
    env = DummyEnv(dt=dt, timing=timing)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No trials were sampled.*")
        stats = env.trial_length_stats(num_trials=1000)

    for key, expected in expected_stats.items():
        assert np.isclose(stats[key], expected, atol=1), f"{key} = {stats[key]} not close to {expected}"


def test_string_methods():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")

        env = ngym.make("AntiReach-v0")
        print(env)  # noqa: T201
        print(env_string(env))  # noqa: T201
        assert str(env) == "<OrderEnforcing<AntiReach>>"
