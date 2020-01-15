"""Reaching to target."""

import numpy as np
from gym import spaces

import neurogym as ngym
from neurogym.ops import tasktools


# TODO: Ground truth and action have different space, making it difficult for SL and RL to work together
class Reaching1D(ngym.EpochEnv):
    metadata = {
        'paper_link': None,
        'paper_name': None,
        'default_timing': {
            'fixation': ('constant', 500),
            'reach': ('constant', 500)},
    }

    def __init__(self, dt=100, timing=None):
        super().__init__(dt=dt, timing=timing)
        # Input noise
        self.sigma = np.sqrt(2 * 100 * 0.01)
        self.sigma_dt = self.sigma / np.sqrt(self.dt)

        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_FAIL = 0.
        self.R_MISS = 0.
        self.abort = False
        # action and observation spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(32,),
                                            dtype=np.float32)
        self.theta = np.arange(0, 2*np.pi, 2*np.pi/16)
        self.state = np.pi

    def new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        self.state = np.pi
        self.trial = {
            'ground_truth': self.rng.uniform(0, np.pi*2)
        }
        self.trial.update(kwargs)
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        self.add_epoch('fixation', after=0)
        self.add_epoch('reach', after='fixation', last_epoch=True)

        ob = self.view_ob('reach')

        ob[:, :16] = np.cos(self.theta - self.trial['ground_truth'])

        self.set_groundtruth('fixation', np.pi)
        self.set_groundtruth('reach', self.trial['ground_truth'])

    def _step(self, action):
        ob = self.obs_now
        ob[16:] = np.cos(self.theta - self.state)
        if action == 1:
            self.state += 0.05
        elif action == 2:
            self.state -= 0.05
        self.state = np.mod(self.state, 2*np.pi)

        gt = self.gt_now
        if self.in_epoch('fixation'):
            reward = 0
        else:
            reward = np.max((1 - np.abs(self.state - gt), -0.1))

        return self.obs_now, reward, False, {'new_trial': False, 'gt': gt}


if __name__ == '__main__':
    from neurogym.tests.test_env import test_run
    env = Reaching1D()
    test_run(env)
    tasktools.plot_struct(env)