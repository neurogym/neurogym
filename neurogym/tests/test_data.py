"""Test Dataset for supervised learning.

All tests in this file can be run by running in command line
pytest test_data.py
"""

import pytest

import numpy as np

import gym
import neurogym as ngym


# Get all supervised learning environment
SLENVS = ngym.all_envs(tag='supervised')


def _test_env(env):
    """Test if one environment can at least be run with Dataset."""
    batch_size = 32
    seq_len = 40
    dataset = ngym.Dataset(
        env, env_kwargs={'dt': 100}, batch_size=batch_size, seq_len=seq_len)
    for i in range(2):
        inputs, target = dataset()
        assert inputs.shape[0] == seq_len
        assert inputs.shape[1] == batch_size
        assert target.shape[0] == seq_len
        assert target.shape[1] == batch_size

    return inputs, target


def test_registered_env():
    """Test if all environments can at least be run."""
    for env_name in sorted(SLENVS):
        print(env_name)
        _test_env(env_name)


def _test_examples_different(env):
    """Test that each example in a batch is different."""
    batch_size = 32
    # need to be long enough to make sure variability in inputs or target
    seq_len = 1000
    dataset = ngym.Dataset(
        env, batch_size=batch_size, seq_len=seq_len)
    inputs, target = dataset()
    # Average across batch
    batch_mean_inputs = np.mean(inputs, axis=1, keepdims=True)
    batch_mean_target = np.mean(target, axis=1, keepdims=True)

    batch_diff_inputs = inputs - batch_mean_inputs
    batch_diff_target = target - batch_mean_target

    assert np.sum(batch_diff_inputs ** 2) > 0
    assert np.sum(batch_diff_target ** 2) > 0


def test_examples_different_registered_env():
    """Test that each example in a batch is different in registered envs."""
    for env_name in sorted(SLENVS):
        print(env_name)
        _test_examples_different(env_name)


def test_examples_different_made_env():
    """Test that each example in a batch is different in created envs."""
    for env_name in sorted(SLENVS):
        print(env_name)
        env = gym.make(env_name)
        _test_examples_different(env)


def test_examples_different_custom_env():
    """Test that each example in a batch is different in created envs."""
    class TestEnv(ngym.TrialEnv):
        def __init__(self, dt=100):
            super().__init__(dt=dt)
            self.timing = {'fixation': dt, 'go': dt}
            name = {'fixation': 0, 'go': 1}
            self.observation_space = ngym.spaces.Box(
                -np.inf, np.inf, shape=(2,), dtype=np.float32, name=name)
            self.action_space = ngym.spaces.Discrete(2)

        def _new_trial(self, **kwargs):
            trial = dict()
            trial['x'] = self.rng.randint(2)
            self.add_period(['fixation', 'go'])
            self.add_ob(1, period='fixation', where='fixation')
            self.add_ob(trial['x'], period='go', where='go')
            self.set_groundtruth(trial['x'], period='go')

            return trial

        def _step(self, action):
            info = {'new_trial': False}
            if self.in_period('fixation'):
                reward = (action == 0) * 1.0
            else:
                reward = (action == 1) * 1.0
            return self.ob_now, reward, False, info

    env = TestEnv()
    _test_examples_different(env)
