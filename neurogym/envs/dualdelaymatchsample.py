from __future__ import division

import numpy as np
from gym import spaces
import neurogym as ngym


class DualDelayMatchSample(ngym.PeriodEnv):
    r"""Two-item Delay-match-to-sample.

    Two sample stimuli are shown simultaneously. Sample 1 and 2 are tested
    sequentially. Whether sample 1 or 2 is tested first is indicated by a cue.
    """
    metadata = {
        'paper_link': 'https://science.sciencemag.org/content/354/6316/1136',
        'paper_name': '''Reactivation of latent working memories with
        transcranial magnetic stimulation''',
        'tags': ['perceptual', 'working memory', 'two-alternative',
                 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0):
        super().__init__(dt=dt)
        self.choices = [1, 2]
        self.cues = [0, 1]

        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': ('constant', 500),
            'sample': ('constant', 500),
            'delay1': ('constant', 500),
            'cue1': ('constant', 500),
            'test1': ('constant', 500),
            'delay2': ('constant', 500),
            'cue2': ('constant', 500),
            'test2': ('constant', 500)}
        if timing:
            self.timing.update(timing)

        self.abort = False

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,),
                                            dtype=np.float32)
        self.ob_dict = {'fixation': 0, 'stimulus1': range(1, 3),
                        'stimulus2': range(3, 5), 'cue1': 5, 'cue2': 6}
        self.action_space = spaces.Discrete(3)
        self.act_dict = {'fixation': 0, 'match': 1, 'non-match': 2}

    def _new_trial(self, **kwargs):
        self.trial = {
            'ground_truth1': self.rng.choice(self.choices),
            'ground_truth2': self.rng.choice(self.choices),
            'sample1': self.rng.choice([0, 0.5]),
            'sample2': self.rng.choice([0, 0.5]),
            'test_order': self.rng.choice([0, 1]),
        }
        self.trial.update(kwargs)

        ground_truth1 = self.trial['ground_truth1']
        ground_truth2 = self.trial['ground_truth2']
        sample1 = self.trial['sample1']
        sample2 = self.trial['sample2']

        test1 = sample1 if ground_truth1 == 1 else 0.5 - sample1
        test2 = sample2 if ground_truth2 == 1 else 0.5 - sample2
        self.trial['test1'] = test1
        self.trial['test2'] = test2

        if self.trial['test_order'] == 0:
            stim_test1_period, stim_test2_period = 'test1', 'test2'
            cue1_period, cue2_period = 'cue1', 'cue2'
        else:
            stim_test1_period, stim_test2_period = 'test2', 'test1'
            cue1_period, cue2_period = 'cue2', 'cue1'

        sample_theta, test_theta = sample1 * np.pi, test1 * np.pi
        stim_sample1 = [np.cos(sample_theta), np.sin(sample_theta)]
        stim_test1 = [np.cos(test_theta), np.sin(test_theta)]

        sample_theta, test_theta = sample2 * np.pi, test2 * np.pi
        stim_sample2 = [np.cos(sample_theta), np.sin(sample_theta)]
        stim_test2 = [np.cos(test_theta), np.sin(test_theta)]

        periods = ['fixation', 'sample', 'delay1', 'cue1', 'test1',
                   'delay2', 'cue2', 'test2']
        self.add_period(periods)

        self.add_ob(1, where='fixation')
        self.add_ob(stim_sample1, 'sample', where='stimulus1')
        self.add_ob(stim_sample2, 'sample', where='stimulus2')
        self.add_ob(1, cue1_period, where='cue1')
        self.add_ob(1, cue2_period, where='cue2')
        self.add_ob(stim_test1, stim_test1_period, where='stimulus1')
        self.add_ob(stim_test2, stim_test2_period, where='stimulus2')
        self.add_randn(0, self.sigma, 'sample')
        self.add_randn(0, self.sigma, 'test1')
        self.add_randn(0, self.sigma, 'test2')

        self.set_groundtruth(ground_truth1, stim_test1_period)
        self.set_groundtruth(ground_truth2, stim_test2_period)

    def _step(self, action):
        new_trial = False
        reward = 0

        obs = self.ob_now
        gt = self.gt_now

        if self.in_period('test1'):
            if action != 0:
                if action == gt:
                    reward = self.rewards['correct']
                    self.performance = 1
                else:
                    reward = self.rewards['fail']
        elif self.in_period('test2'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward = self.rewards['correct']
                    self.performance = 1
                else:
                    reward = self.rewards['fail']
        else:
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt}