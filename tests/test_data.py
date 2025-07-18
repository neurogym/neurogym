"""Test Dataset for supervised learning."""

import warnings

import numpy as np

from neurogym.core import TrialEnv
from neurogym.envs.registration import all_envs, make
from neurogym.utils import spaces
from neurogym.utils.data import Dataset
from neurogym.utils.logging import logger

# Get all supervised learning environment
SL_ENVS = all_envs(tag="supervised")


def _test_env(env):
    """Test if one environment can at least be run with Dataset."""
    batch_size = 32
    seq_len = 40
    dataset = Dataset(
        env,
        env_kwargs={"dt": 100},
        batch_size=batch_size,
        seq_len=seq_len,
    )
    for _ in range(2):
        inputs, target = dataset()
        assert inputs.shape[0] == seq_len
        assert inputs.shape[1] == batch_size
        assert target.shape[0] == seq_len
        assert target.shape[1] == batch_size

    return inputs, target


def test_registered_env():
    """Test if all environments can at least be run."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")
        try:
            for env_name in sorted(SL_ENVS):
                _test_env(env_name)
        except:
            logger.error(f"Failure in: {env_name}")
            raise


def _test_examples_different(env) -> None:
    """Test that each example in a batch is different."""
    batch_size = 32
    # need to be long enough to make sure variability in inputs or target
    seq_len = 1000
    dataset = Dataset(env, batch_size=batch_size, seq_len=seq_len)
    inputs, target = dataset()
    # Average across batch
    batch_mean_inputs = np.mean(inputs, axis=1, keepdims=True)
    batch_mean_target = np.mean(target, axis=1, keepdims=True)

    batch_diff_inputs = inputs - batch_mean_inputs
    batch_diff_target = target - batch_mean_target

    assert np.sum(batch_diff_inputs**2) > 0
    assert np.sum(batch_diff_target**2) > 0


def test_examples_different_registered_env():
    """Test that each example in a batch is different in registered envs."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")
        try:
            for env_name in sorted(SL_ENVS):
                _test_examples_different(env_name)
        except:
            logger.error(f"Failure in: {env_name}")
            raise


def test_examples_different_made_env():
    """Test that each example in a batch is different in created envs."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")
        try:
            for env_name in sorted(SL_ENVS):
                env = make(env_name)
                _test_examples_different(env)
        except:
            logger.error(f"Failure in: {env_name}")
            raise


def test_examples_different_custom_env():
    """Test that each example in a batch is different in created envs."""

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
            trial = {}
            trial["x"] = self.rng.randint(2)
            self.add_period(["fixation", "go"])
            self.add_ob(1, period="fixation", where="fixation")
            self.add_ob(trial["x"], period="go", where="go")
            self.set_groundtruth(trial["x"], period="go")

            return trial

        def _step(self, action):
            info = {"new_trial": False}
            reward = (action == 0) * 1.0 if self.in_period("fixation") else (action == 1) * 1.0
            terminated = False
            truncated = False
            return self.ob_now, reward, terminated, truncated, info

    env = TestEnv()
    _test_examples_different(env)
