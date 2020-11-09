"""Test core.py"""

# import pytest

import numpy as np
import neurogym as ngym


def test_one_step_mismatch():
    """
    Test whether the agent can act as same as the current ob.
    If current ob = 1, the rewarded action = 1;
    If current ob = 0, the rewarded action = 0.
    The agent cannot act correctly when there's a one-step-mismatch problem.
    """

    class TestEnv(ngym.TrialEnv):
        def __init__(self, dt=100):
            super().__init__(dt=dt)
            self.timing = {'go1': dt, 'go2': dt}
            self.observation_space = ngym.spaces.Box(
                -np.inf, np.inf, shape=(1,), dtype=np.float32)
            self.action_space = ngym.spaces.Discrete(2)

        def _new_trial(self, **kwargs):
            periods = ['go1', 'go2']
            self.add_period(periods)
            pe = self.rng.choice(periods)
            self.add_ob(1, period=[pe])
            trial = dict()
            return trial

        def _step(self, action):
            info = {'new_trial': False}
            if action == self.ob_now:
                reward = 1
            else:
                reward = -1
            if self.in_period('go2'):
                info['new_trial'] = True
                ob_next = None # should be replaced by ob from new trial
            else:
                ob_next = self.ob[self.t_ind + 1]
            return ob_next, reward, False, info

    env = TestEnv()
    ob = env.reset(no_step=True)
    for i in range(5):
        ob, rew, done, info = env.step(action=ob)
        print(ob, rew)
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
        assert ob[0] == (i % 5) + 2  # each trial is 5 steps
