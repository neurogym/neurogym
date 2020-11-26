"""Test core.py"""

# import pytest

import numpy as np
import neurogym as ngym


def test_one_step_mismatch():
    """Test the agent gets rewarded if it fixates after seeing fixation cue."""

    class TestEnv(ngym.TrialEnv):
        def __init__(self, dt=100):
            super().__init__(dt=dt)
            self.timing = {'fixation': dt, 'go': dt}
            name = {'fixation': 0, 'go': 1}
            self.observation_space = ngym.spaces.Box(
                -np.inf, np.inf, shape=(2,), dtype=np.float32, name=name)
            self.action_space = ngym.spaces.Discrete(2)

        def _new_trial(self, **kwargs):
            self.add_period(['fixation', 'go'])
            self.add_ob(1, period='fixation', where='fixation')
            self.add_ob(1, period='go', where='go')
            trial = dict()
            return trial

        def _step(self, action):
            info = {'new_trial': False}
            if self.in_period('fixation'):
                reward = (action == 0) * 1.0
            else:
                reward = (action == 1) * 1.0
            return self.ob_now, reward, False, info

    env = TestEnv()
    ob = env.reset(no_step=True)
    for i in range(10):
        action = np.argmax(ob)  # fixate if observes fixation input, vice versa
        ob, rew, done, info = env.step(action=action)
        if i > 0:
            assert rew == 1


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
        assert ob[0] == ((i + 1) % 5) + 1  # each trial is 5 steps
