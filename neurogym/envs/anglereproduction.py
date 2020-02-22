"""Reproducing angles."""

import numpy as np
from gym import spaces
from neurogym.utils import tasktools
import neurogym as ngym


class AngleReproduction(ngym.PeriodEnv):
    metadata = {
        'description': 'The agent has to reproduce to two angles ' +
        'separated by a constant delay.',
        'paper_link': 'https://www.pnas.org/content/114/43/E9115.short',
        'paper_name': """Visual perception as retrospective Bayesian
        decoding from high- to low-level features""",
        'timing': {  # TODO: Timing not from paper yet
            'fixation': ('constant', 500),
            'stim1': ('constant', 500),
            'delay1': ('constant', 500),
            'stim2': ('constant', 500),
            'delay2': ('constant', 500),
            'go1': ('constant', 500),
            'go2': ('constant', 500)},
        'tags': ['perceptual', 'working memory', 'delayed response',
                 'steps action space']
    }

    def __init__(self, dt=100, rewards=None, timing=None):
        """
        The agent has to reproduce to two angles separated by a constant delay.
        dt: Timestep duration. (def: 100 (ms), int)
        rewards:
            R_CORRECT: given when correct. (def: +1., float)
            R_FAIL: given when incorrect. (def: -0.1, float)
        timing: Description and duration of periods forming a trial.
        """
        super().__init__(dt=dt, timing=timing)
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
        # Periods
        # ---------------------------------------------------------------------
        periods = ['fixation', 'stim1', 'delay1', 'stim2',
                   'delay2', 'go1', 'go2']
        self.add_period(periods[0], after=0)
        for i in range(1, len(periods)):
            self.add_period(periods[i], after=periods[i - 1],
                            last_period=i == len(periods) - 1)

        ob = self.view_ob('stim1')
        ob[:, :16] = np.cos(self.theta - self.trial['ground_truth1'])
        ob = self.view_ob('stim2')
        ob[:, :16] = np.cos(self.theta - self.trial['ground_truth2'])
        ob = self.view_ob('go1')
        ob[:, 32] = 1
        ob = self.view_ob('go2')
        ob[:, 33] = 1

        self.set_groundtruth('go1', self.trial['ground_truth1'])
        self.set_groundtruth('go2', self.trial['ground_truth2'])
        self.dec_per_dur = (self.end_ind['go1'] - self.start_ind['go1']) +\
            (self.end_ind['go2'] - self.start_ind['go2'])

    def _step(self, action):
        ob = self.obs_now
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
