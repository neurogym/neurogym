"""Hierarchical reasoning tasks."""

import numpy as np
from gym import spaces

import neurogym as ngym


class HierarchicalReasoning(ngym.TrialEnv):
    """Hierarchical reasoning of rules.


    """
    metadata = {
        'paper_link': 'https://science.sciencemag.org/content/364/6441/eaav8911',
        'paper_name': "Hierarchical reasoning by neural circuits in the frontal cortex",
        'tags': ['perceptual', 'two-alternative', 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None):
        super().__init__(dt=dt)
        self.choices = [0, 1]

        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': ('truncated_exponential', [600, 400, 800]),
            'rule_target': ('constant', 1000),
            'fixation2': ('truncated_exponential', [600, 400, 900]),
            'flash1': ('constant', 100),
            'delay': ('choice', [530, 610, 690, 770, 850, 930, 1010, 1090, 1170]),
            'flash2': ('constant', 100),
            'decision': ('constant', 700),
        }
        if timing:
            self.timing.update(timing)
        self.mid_delay = np.median(self.timing['delay'][1])

        self.abort = False

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(5,), dtype=np.float32)
        self.ob_dict = {'fixation': 0, 'rule': [1, 2], 'stimulus': [3, 4]}
        self.action_space = spaces.Discrete(5)
        self.act_dict = {'fixation': 0, 'rule': [1, 2], 'choice': [3, 4]}

        self.chose_correct_rule = False
        self.rule = 0
        self.trial_in_block = 0
        self.block_size = 10
        self.new_block()

    def new_block(self):
        self.block_size = self.rng.random_integers(10, 20)
        self.rule = 1 - self.rule  # alternate rule
        self.trial_in_block = 0

    def _new_trial(self, **kwargs):
        interval = self.sample_time('delay')
        self.trial = {
            'interval': interval,
            'rule': self.rule,
            'stimulus': self.rng.choice(self.choices)
        }
        self.trial.update(kwargs)

        # Is interval long? When interval == mid_delay, randomly assign
        long_interval = interval > self.mid_delay + (self.rng.rand()-0.5)
        # Is the response pro or anti?
        pro_choice = int(long_interval) == self.trial['rule']
        self.trial['long_interval'] = long_interval
        self.trial['pro_choice'] = pro_choice

        # Periods
        periods = ['fixation', 'rule_target', 'fixation2', 'flash1',
                   'delay', 'flash2', 'decision']
        self.add_period(periods)

        # Observations
        stimulus = self.ob_dict['stimulus'][self.trial['stimulus']]
        if pro_choice:
            choice = self.trial['stimulus']
        else:
            choice = 1 - self.trial['stimulus']

        self.add_ob(1, where='fixation')
        self.set_ob(0, 'decision', where='fixation')
        self.add_ob(1, 'rule_target', where='rule')
        self.add_ob(1, 'flash1', where=stimulus)
        self.add_ob(1, 'flash2', where=stimulus)

        # Ground truth
        self.set_groundtruth(self.act_dict['choice'][choice], 'decision')
        self.set_groundtruth(
            self.act_dict['rule'][self.trial['rule']], 'rule_target')

        # Start new block?
        self.trial_in_block += 1
        if self.trial_in_block >= self.block_size:
            self.new_block()

    def _step(self, action):
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now
        # observations
        if self.in_period('decision'):
            if action != 0:
                new_trial = True
                if (action == gt) and self.chose_correct_rule:
                    reward += self.rewards['correct']
                    self.performance = 1
                else:
                    reward += self.rewards['fail']
        elif self.in_period('rule_target'):
            self.chose_correct_rule = (action == gt)
        else:
            if action != 0:  # action = 0 means fixating
                new_trial = self.abort
                reward += self.rewards['abort']

        if new_trial:
            self.chose_correct_rule = False

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}
