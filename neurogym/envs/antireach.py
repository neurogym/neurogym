"""Anti-reach or anti-saccade task."""

import numpy as np
from gym import spaces

import neurogym as ngym
from neurogym.utils import tasktools


class AntiReach1D(ngym.PeriodEnv):
    metadata = {
        'description': 'The agent has to move in the direction opposite ' +
        'to the one indicated by the observation.',
        'paper_link': 'https://www.nature.com/articles/nrn1345',
        'paper_name': """Look away: the anti-saccade task and
        the voluntary control of eye movement""",
        'tags': ['perceptual', 'steps action space']
    }

    def __init__(self, dt=100, rewards=None, timing=None):
        """
        The agent has to move in the direction opposite to the one indicated
        by the observation.
        """
        super().__init__(dt=dt)
        self.contexts = [0, 1]
        # action and observation spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(32+2,),
                                            dtype=np.float32)
        self.theta = np.arange(0, 2*np.pi, 2*np.pi/16)
        self.state = np.pi

        # Rewards
        self.rewards = {'correct': +1., 'fail': -0.1}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': ('constant', 500),
            'reach': ('constant', 500)}
        if timing:
            self.timing.update(timing)

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
        # Periods
        # ---------------------------------------------------------------------
        periods = ['fixation', 'reach']
        self.add_period(periods, after=0, last_period=True)

        ob = self.view_ob('fixation')
        ob[:, 32 + self.trial['context']] += 1

        ob = self.view_ob('reach')
        # Actual stimulus pi away
        shift = 0 if self.trial['context'] == 0 else np.pi
        ob[:, :16] = np.cos(self.theta - (self.trial['ground_truth'] + shift))

        self.set_groundtruth(np.pi, 'fixation')
        self.set_groundtruth(self.trial['ground_truth'], 'reach')

    def _step(self, action):
        ob = self.obs_now
        ob[16:32] = np.cos(self.theta - self.state)
        if action == 1:
            self.state += 0.05
        elif action == 2:
            self.state -= 0.05
        self.state = np.mod(self.state, 2*np.pi)

        gt = self.gt_now
        if self.in_period('fixation'):
            reward = 0
        else:
            reward =\
                np.max((self.rewards['correct']-tasktools.circular_dist(self.state-gt),
                        self.rewards['fail']))

        return ob, reward, False, {'new_trial': False}


if __name__ == '__main__':
    from neurogym.tests import test_run
    env = AntiReach1D()
    test_run(env)
    ngym.utils.plot_env(env)
