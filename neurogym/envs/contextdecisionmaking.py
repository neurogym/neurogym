from __future__ import division

import numpy as np

import neurogym as ngym
from neurogym import spaces


class SingleContextDecisionMaking(ngym.TrialEnv):
    """Context-dependent decision-making task.

    The agent simultaneously receives stimulus inputs from two modalities (
    for example, a colored random dot motion pattern with color and motion
    modalities). The agent needs to make a perceptual decision based on only
    one of the two modalities, while ignoring the other. The agent reports
    its decision during the decision period, with an optional delay period
    in between the stimulus period and the decision period. The relevant
    modality is not explicitly signaled.

    Args:
        context: int, 0 or 1 for the two context (rules). If 0, need to
            focus on modality 0 (the first one)
    """
    metadata = {
        'paper_link': 'https://www.nature.com/articles/nature12742',
        'paper_name': '''Context-dependent computation by recurrent
         dynamics in prefrontal cortex''',
        'tags': ['perceptual', 'context dependent', 'two-alternative',
                 'supervised']
    }

    def __init__(self, dt=100, context=0, rewards=None, timing=None,
                 sigma=1.0, dim_ring=2):
        super().__init__(dt=dt)

        # trial conditions
        self.choices = [1, 2]  # left, right choice
        self.cohs = [5, 15, 50]
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise
        self.context = context

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': 300,
            # 'target': 350,
            'stimulus': 750,
            'delay': ngym.random.TruncExp(600, 300, 3000),
            'decision': 100}
        if timing:
            self.timing.update(timing)

        self.abort = False

        # set action and observation space
        self.theta = np.linspace(0, 2 * np.pi, dim_ring + 1)[:-1]
        self.choices = np.arange(dim_ring)

        name = {
            'fixation': 0,
            'stimulus_mod1': range(1, dim_ring + 1),
            'stimulus_mod2': range(dim_ring + 1, 2 * dim_ring + 1)}
        shape = (1 + 2 * dim_ring,)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=shape, dtype=np.float32, name=name)

        name = {'fixation': 0, 'choice': range(1, dim_ring+1)}
        self.action_space = spaces.Discrete(1+dim_ring, name=name)

    def _new_trial(self, **kwargs):
        # Trial
        trial = {
            'ground_truth': self.rng.choice(self.choices),
            'other_choice': self.rng.choice(self.choices),
            'context': self.context,
            'coh_0': self.rng.choice(self.cohs),
            'coh_1': self.rng.choice(self.cohs),
        }
        trial.update(kwargs)

        choice_0, choice_1 =\
            trial['ground_truth'], trial['other_choice']
        if trial['context'] == 1:
            choice_1, choice_0 = choice_0, choice_1
        coh_0, coh_1 = trial['coh_0'], trial['coh_1']

        stim_theta_0 = self.theta[choice_0]
        stim_theta_1 = self.theta[choice_1]
        ground_truth = trial['ground_truth']

        # Periods
        periods = ['fixation', 'stimulus', 'delay', 'decision']
        self.add_period(periods)

        self.add_ob(1, where='fixation')
        stim = np.cos(self.theta - stim_theta_0) * (coh_0 / 200) + 0.5
        self.add_ob(stim, 'stimulus', where='stimulus_mod1')
        stim = np.cos(self.theta - stim_theta_1) * (coh_1 / 200) + 0.5
        self.add_ob(stim, 'stimulus', where='stimulus_mod2')
        self.add_randn(0, self.sigma, 'stimulus')
        self.set_ob(0, 'decision')

        self.set_groundtruth(ground_truth, period='decision', where='choice')

        return trial

    def _step(self, action):
        ob = self.ob_now
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

        return ob, reward, False, {'new_trial': new_trial, 'gt': gt}


class ContextDecisionMaking(ngym.TrialEnv):
    """Context-dependent decision-making task.

    The agent simultaneously receives stimulus inputs from two modalities (
    for example, a colored random dot motion pattern with color and motion
    modalities). The agent needs to make a perceptual decision based on
    only one of the two modalities, while ignoring the other. The relevant
    modality is explicitly indicated by a rule signal.
    """
    metadata = {
        'paper_link': 'https://www.nature.com/articles/nature12742',
        'paper_name': '''Context-dependent computation by recurrent
         dynamics in prefrontal cortex''',
        'tags': ['perceptual', 'context dependent', 'two-alternative',
                 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0):
        super().__init__(dt=dt)

        # trial conditions
        self.contexts = [0, 1]  # index for context inputs
        self.choices = [1, 2]  # left, right choice
        self.cohs = [5, 15, 50]
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': 300,
            # 'target': 350,
            'stimulus': 750,
            'delay': ngym.random.TruncExp(600, 300, 3000),
            'decision': 100}
        if timing:
            self.timing.update(timing)

        self.abort = False

        # set action and observation space
        names = ['fixation', 'stim1_mod1', 'stim2_mod1',
                 'stim1_mod2', 'stim2_mod2', 'context1', 'context2']
        name = {name: i for i, name in enumerate(names)}
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,),
                                            dtype=np.float32, name=name)

        name = {'fixation': 0, 'choice1': 1, 'choice2': 2}
        self.action_space = spaces.Discrete(3, name=name)

    def _new_trial(self, **kwargs):
        # -------------------------------------------------------------------------
        # Trial
        # -------------------------------------------------------------------------
        trial = {
            'ground_truth': self.rng.choice(self.choices),
            'other_choice': self.rng.choice(self.choices),
            'context': self.rng.choice(self.contexts),
            'coh_0': self.rng.choice(self.cohs),
            'coh_1': self.rng.choice(self.cohs),
        }
        trial.update(kwargs)

        choice_0, choice_1 =\
            trial['ground_truth'], trial['other_choice']
        if trial['context'] == 1:
            choice_1, choice_0 = choice_0, choice_1
        coh_0, coh_1 = trial['coh_0'], trial['coh_1']

        signed_coh_0 = coh_0 if choice_0 == 1 else -coh_0
        signed_coh_1 = coh_1 if choice_1 == 1 else -coh_1
        # -----------------------------------------------------------------------
        # Periods
        # -----------------------------------------------------------------------
        periods = ['fixation', 'stimulus', 'delay', 'decision']
        self.add_period(periods)

        self.add_ob(1, where='fixation')
        self.add_ob((1 + signed_coh_0 / 100) / 2, period='stimulus', where='stim1_mod1')
        self.add_ob((1 - signed_coh_0 / 100) / 2, period='stimulus', where='stim2_mod1')
        self.add_ob((1 + signed_coh_1 / 100) / 2, period='stimulus', where='stim1_mod2')
        self.add_ob((1 - signed_coh_1 / 100) / 2, period='stimulus', where='stim2_mod2')
        self.add_randn(0, self.sigma, 'stimulus')
        self.set_ob(0, 'decision')

        if trial['context'] == 0:
            self.add_ob(1, where='context1')
        else:
            self.add_ob(1, where='context2')

        self.set_groundtruth(trial['ground_truth'], 'decision')

        return trial

    def _step(self, action):
        ob = self.ob_now
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

        return ob, reward, False, {'new_trial': new_trial, 'gt': gt}
