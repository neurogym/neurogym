"""Reproducing angles."""

import numpy as np
from gym import spaces

import neurogym as ngym
from neurogym.ops import tasktools


class AngleReproduction(ngym.EpochEnv):
    metadata = {
        'paper_link': 'https://www.pnas.org/content/114/43/E9115.short',
        'paper_name': """Visual perception as retrospective Bayesian 
        decoding from high- to low-level features""",
        'default_timing': {  # TODO: Timing not from paper yet
            'fixation': ('constant', 500),
            'stim1': ('constant', 500),
            'delay1': ('constant', 500),
            'stim2': ('constant', 500),
            'delay2': ('constant', 500),
            'go1': ('constant', 500),
            'go2': ('constant', 500),
        },
    }

    def __init__(self, dt=100, timing=None):
        super().__init__(dt=dt, timing=timing)
        # action and observation spaces
        # Do nothing, rotate clockwise, rotatet counterclockwise
        self.action_space = spaces.Discrete(3)
        # 0-31 is angle, 32 go1, 33 go2
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(34,),
                                            dtype=np.float32)
        self.theta = np.arange(0, 2*np.pi, 2*np.pi/32)
        self.state = np.pi

    def new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        self.state = np.pi
        self.trial = {
            'ground_truth1': self.rng.uniform(0, np.pi * 2),
            'ground_truth2': self.rng.uniform(0, np.pi * 2)
        }
        self.trial.update(kwargs)
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        epochs = ['fixation', 'stim1', 'delay1', 'stim2', 'delay2', 'go1', 'go2']
        self.add_epoch(epochs[0], after=0)
        for i in range(1, len(epochs)):
            self.add_epoch(epochs[i], after=epochs[i - 1], last_epoch=i == len(epochs) - 1)

        ob = self.view_ob('stim1')
        ob[:, :32] = np.cos(self.theta - self.trial['ground_truth1'])
        ob = self.view_ob('stim2')
        ob[:, :32] = np.cos(self.theta - self.trial['ground_truth2'])
        ob = self.view_ob('go1')
        ob[:, 32] = 1
        ob = self.view_ob('go2')
        ob[:, 33] = 1

        self.set_groundtruth('go1', self.trial['ground_truth1'])
        self.set_groundtruth('go2', self.trial['ground_truth2'])

    def _step(self, action):
        ob = self.obs_now
        if action == 1:
            self.state += 0.05
        elif action == 2:
            self.state -= 0.05

        self.state = np.mod(self.state, 2*np.pi)

        gt = self.gt_now
        reward = 0
        if self.in_epoch('go1') or self.in_epoch('go2'):
            reward = np.max((1 - np.abs(self.state - gt), -0.1))

        return ob, reward, False, {'new_trial': False, 'gt': gt}


if __name__ == '__main__':
    from neurogym.tests import test_run
    env = AngleReproduction()
    test_run(env)
    tasktools.plot_struct(env)