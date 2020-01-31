"""Anti-reach or anti-saccade task."""

import numpy as np
from gym import spaces

import neurogym as ngym
from neurogym.meta import info
from neurogym.ops import tasktools


class AntiReach1D(ngym.EpochEnv):
    metadata = {
        'description': 'The agent has to move in the direction opposite ' +
        'to the one indicated by the observation.',
        'paper_link': 'https://www.nature.com/articles/nrn1345',
        'paper_name': """Look away: the anti-saccade task and
        the voluntary control of eye movement""",
        'timing': {
            'fixation': ('constant', 500),
            'reach': ('constant', 500)},
        'tags': ['perceptual', 'continuous action space', 'supervised']
    }

    def __init__(self, dt=100, timing=None):
        super().__init__(dt=dt, timing=timing)
        self.contexts = [0, 1]
        # action and observation spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(32+2,),
                                            dtype=np.float32)
        self.theta = np.arange(0, 2*np.pi, 2*np.pi/16)
        self.state = np.pi

    def new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        self.state = np.pi
        self.trial = {
            'ground_truth': self.rng.uniform(0, np.pi*2),
            'context': self.rng.choice(self.contexts)
        }
        self.trial.update(kwargs)
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        self.add_epoch('fixation', after=0)
        self.add_epoch('reach', after='fixation', last_epoch=True)

        ob = self.view_ob('fixation')
        ob[:, 32 + self.trial['context']] += 1

        ob = self.view_ob('reach')
        # Actual stimulus pi away
        shift = 0 if self.trial['context'] == 0 else np.pi
        ob[:, :16] = np.cos(self.theta - (self.trial['ground_truth'] + shift))

        self.set_groundtruth('fixation', np.pi)
        self.set_groundtruth('reach', self.trial['ground_truth'])

    def _step(self, action):
        ob = self.obs_now
        ob[16:32] = np.cos(self.theta - self.state)
        if action == 1:
            self.state += 0.05
        elif action == 2:
            self.state -= 0.05
        self.state = np.mod(self.state, 2*np.pi)

        gt = self.gt_now
        if self.in_epoch('fixation'):
            reward = 0
        else:
            reward =\
                np.max((1 - tasktools.circular_dist(self.state - gt), -0.1))

        return ob, reward, False, {'new_trial': False, 'gt': gt}


if __name__ == '__main__':
    from neurogym.tests import test_run
    env = AntiReach1D()
    test_run(env)
    info.plot_struct(env)
