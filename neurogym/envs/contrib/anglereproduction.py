"""Reproducing angles."""

import numpy as np
from gym import spaces
from neurogym.utils import tasktools
import neurogym as ngym


class AngleReproduction(ngym.TrialEnv):
    r"""Angle reproduction.

    The agent has to reproduce to two angles separated
    by a constant delay.
    """
    metadata = {
        'paper_link': 'https://www.pnas.org/content/114/43/E9115.short',
        'paper_name': """Visual perception as retrospective Bayesian
        decoding from high- to low-level features""",
        'tags': ['perceptual', 'working memory', 'delayed response',
                 'steps action space']
    }

    def __init__(self, dt=100, rewards=None, timing=None):
        super().__init__(dt=dt)
        # action and observation spaces
        # Do nothing, rotate clockwise, rotatet counterclockwise
        self.action_space = spaces.Discrete(3)
        # 0-31 is angle, 32 go1, 33 go2
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(34,),
                                            dtype=np.float32)
        self.theta = np.arange(0, 2*np.pi, 2*np.pi/16)
        self.state = np.pi

        # Rewards
        self.rewards = {'correct': +1., 'fail': -0.1}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': 500,
            'stim1': 500,
            'delay1': 500,
            'stim2': 500,
            'delay2': 500,
            'go1': 500,
            'go2': 500}
        if timing:
            self.timing.update(timing)

    def _new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        self.state = np.pi
        trial = {
            'ground_truth1': self.rng.uniform(0, np.pi * 2),
            'ground_truth2': self.rng.uniform(0, np.pi * 2)
        }
        trial.update(kwargs)
        # ---------------------------------------------------------------------
        # Periods
        # ---------------------------------------------------------------------
        periods = ['fixation', 'stim1', 'delay1', 'stim2',
                   'delay2', 'go1', 'go2']
        self.add_period(periods)

        ob = self.view_ob('stim1')
        ob[:, :16] = np.cos(self.theta - trial['ground_truth1'])
        ob = self.view_ob('stim2')
        ob[:, :16] = np.cos(self.theta - trial['ground_truth2'])
        ob = self.view_ob('go1')
        ob[:, 32] = 1
        ob = self.view_ob('go2')
        ob[:, 33] = 1

        self.set_groundtruth(trial['ground_truth1'], 'go1')
        self.set_groundtruth(trial['ground_truth2'], 'go2')
        self.dec_per_dur = (self.end_ind['go1'] - self.start_ind['go1']) +\
            (self.end_ind['go2'] - self.start_ind['go2'])

        return trial

    def _step(self, action):
        ob = self.ob_now
        ob[16:32] = np.cos(self.theta - self.state)
        if action == 1:
            self.state += 0.05
        elif action == 2:
            self.state -= 0.05

        self.state = np.mod(self.state, 2*np.pi)

        gt = self.gt_now
        reward = 0
        if self.in_period('go1') or self.in_period('go2'):
            reward =\
                np.max((self.rewards['correct']-tasktools.circular_dist(self.state-gt),
                        self.rewards['fail']))
            norm_rew = (reward-self.rewards['fail'])/(self.rewards['correct']-self.rewards['fail'])
            self.performance += norm_rew/self.dec_per_dur

        return ob, reward, False, {'new_trial': False}


if __name__ == '__main__':
    from neurogym.tests import test_run
    env = AngleReproduction()
    test_run(env)
    ngym.utils.plot_env(env)
