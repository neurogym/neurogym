"""Reaching to target."""

import numpy as np
from gym import spaces

import neurogym as ngym
from neurogym.meta import info
from neurogym.ops import tasktools


# TODO: Ground truth and action have different space,
# making it difficult for SL and RL to work together
class Reaching1D(ngym.EpochEnv):
    metadata = {
        'description': 'The agent has to reproduce the angle indicated' +
        ' by the observation.',
        'paper_link': 'https://science.sciencemag.org/content/233/4771/1416',
        'paper_name': 'Neuronal population coding of movement direction',
        'timing': {
            'fixation': ('constant', 500),
            'reach': ('constant', 500)},
        'tags': ['motor', 'continuous action space', 'supervised']
    }

    def __init__(self, dt=100, timing=None):
        super().__init__(dt=dt, timing=timing)
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
            reward =\
                np.max((1 - tasktools.circular_dist(self.state - gt), -0.1))

        return ob, reward, False, {'new_trial': False, 'gt': gt}


class Reaching1DWithSelfDistraction(ngym.EpochEnv):
    """
    Reaching with self distraction.
    In this task, the reaching state itself generates strong inputs that
    overshadows the actual target input. This task is inspired by behavior
    in electric fish where the electric sensing organ is distracted by
    discharges from its own electric organ for active sensing.
    Similar phenomena in bats.
    """
    metadata = {
        'description': '''The agent has to reproduce the angle indicated
         by the observation. Furthermore, the reaching state itself
         generates strong inputs that overshadows the actual target input.''',
        'paper_link': None,
        'paper_name': None,
        'timing': {
            'fixation': ('constant', 500),
            'reach': ('constant', 500)},
        'tags': ['motor', 'continuous action space', 'supervised']
    }

    def __init__(self, dt=100, timing=None):
        super().__init__(dt=dt, timing=timing)

        # action and observation spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(32,),
                                            dtype=np.float32)
        self.theta = np.arange(0, 2*np.pi, 2*np.pi/32)
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
        # Signal is weaker than the self-distraction
        ob += np.cos(self.theta - self.trial['ground_truth']) * 0.3

        self.set_groundtruth('fixation', np.pi)
        self.set_groundtruth('reach', self.trial['ground_truth'])

    def _step(self, action):
        ob = self.obs_now + np.cos(self.theta - self.state)
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
    # env = Reaching1D()
    env = Reaching1DWithSelfDistraction()
    test_run(env)
    info.plot_struct(env)
