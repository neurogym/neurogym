"""Test core.py"""

import pytest

import numpy as np
import neurogym as ngym


def test_addob_instep():
    """Test if we can add observation during step."""

    class TestEnv(ngym.TrialEnv):
        def __init__(self, dt=100):
            super().__init__(dt=dt)
            self.timing = {'go': 500}
            self.observation_space = ngym.spaces.Box(
                -np.inf, np.inf, shape=(1,), dtype=np.float32)
            self.action_space = ngym.spaces.Discrete(3)

        def _new_trial(self, **kwargs):
            trial = dict()
            self.add_period('go')
            self.add_ob(1)
            return trial

        def _step(self, action):
            new_trial = False
            reward = 0
            self.add_ob(1)
            return self.ob_now, reward, False, {'new_trial': new_trial}

    env = TestEnv()
    env.reset(no_step=True)
    for i in range(10):
        ob, rew, done, info = env.step(action=0)
        assert ob[0] == (i % 5) + 2   # each trial is 5 steps
