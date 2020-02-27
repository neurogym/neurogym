"""
Context-dependent integration task, based on
  Context-dependent computation by recurrent dynamics in prefrontal cortex.
  V Mante, D Sussillo, KV Shinoy, & WT Newsome, Nature 2013.
  http://dx.doi.org/10.1038/nature12742

"""
from __future__ import division

import numpy as np
from gym import spaces
import neurogym as ngym
from neurogym.utils.inputs import GaussianNoise


class ContextDecisionMaking(ngym.PeriodEnv):
    metadata = {
        'description': 'Agent has to perform one of two different perceptual' +
        ' discriminations. On every trial, a contextual cue indicates which' +
        ' one to perform.',
        'paper_link': 'https://www.nature.com/articles/nature12742',
        'paper_name': '''Context-dependent computation by recurrent
         dynamics in prefrontal cortex''',
        'tags': ['perceptual', 'context dependent', 'two-alternative',
                 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None):
        """
        Agent has to perform one of two different perceptual discriminations.
        On every trial, a contextual cue indicates which one to perform.
        """
        super().__init__(dt=dt)

        # trial conditions
        self.contexts = [0, 1]  # index for context inputs
        self.choices = [1, 2]  # left, right choice
        self.cohs = [5, 15, 50]

        # Input noise
        sigma = np.sqrt(2*100*0.02)
        self.sigma_dt = sigma/np.sqrt(self.dt)

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': ('constant', 300),
            # 'target': ('constant', 350),  # TODO: not implemented
            'stimulus': ('constant', 750),
            'delay': ('truncated_exponential', [600, 300, 3000]),
            'decision': ('constant', 100)}  # XXX: not specified
        if timing:
            self.timing.update(timing)

        self.abort = False

        # set action and observation space
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,),
                                            dtype=np.float32)
        names = ['fixation', 'stim1_mod1', 'stim2_mod1',
                 'stim1_mod2', 'stim2_mod2', 'context1', 'context2']
        self.ob_dict = {name: i for i, name in enumerate(names)}

    def new_trial(self, **kwargs):
        # -------------------------------------------------------------------------
        # Trial
        # -------------------------------------------------------------------------
        self.trial = {
            'ground_truth': self.rng.choice(self.choices),
            'other_choice': self.rng.choice(self.choices),
            'context': self.rng.choice(self.contexts),
            'coh_0': self.rng.choice(self.cohs),
            'coh_1': self.rng.choice(self.cohs),
        }
        self.trial.update(kwargs)

        choice_0, choice_1 =\
            self.trial['ground_truth'], self.trial['other_choice']
        if self.trial['context'] == 1:
            choice_1, choice_0 = choice_0, choice_1
        coh_0, coh_1 = self.trial['coh_0'], self.trial['coh_1']

        signed_coh_0 = coh_0 if choice_0 == 1 else -coh_0
        signed_coh_1 = coh_1 if choice_1 == 1 else -coh_1
        # -----------------------------------------------------------------------
        # Periods
        # -----------------------------------------------------------------------
        periods = ['fixation', 'stimulus', 'delay', 'decision']
        self.add_period(periods, after=0, last_period=True)

        self.add_ob(1, where='fixation')
        self.add_ob((1 + signed_coh_0 / 100) / 2, period='stimulus', where='stim1_mod1')
        self.add_ob((1 - signed_coh_0 / 100) / 2, period='stimulus', where='stim2_mod1')
        self.add_ob((1 + signed_coh_1 / 100) / 2, period='stimulus', where='stim1_mod2')
        self.add_ob((1 - signed_coh_1 / 100) / 2, period='stimulus', where='stim2_mod2')
        self.add_randn(0, self.sigma_dt, 'stimulus')
        self.set_ob(0, 'decision')

        if self.trial['context'] == 0:
            self.add_ob(1, where='context1')
        else:
            self.add_ob(1, where='context2')

        self.set_groundtruth(self.trial['ground_truth'], 'decision')

    def _step(self, action):
        obs = self.obs_now
        gt = self.gt_now

        new_trial = False
        reward = 0
        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:  # broke fixation
                new_trial = True
                if action == gt:
                    reward = self.rewards['correct']
                    self.performance = 1

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt}


if __name__ == '__main__':
    env = ContextDecisionMaking()
    env.seed(seed=0)
    ngym.utils.plot_env(env, num_steps_env=100, def_act=0)
    env = ContextDecisionMaking()
    env.seed(seed=0)
    ngym.utils.plot_env(env, num_steps_env=100, def_act=0)
