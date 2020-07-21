#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from gym import spaces
import neurogym as ngym


class nalt_PerceptualDecisionMaking(ngym.PeriodEnv):
    """N-alternative perceptual decision-making.

    N-alternative forced choice task in which the subject has
    to integrate N stimuli to decide which one is higher on average.

    Args:
        stim_scale: Controls the difficulty of the experiment. (def: 1., float)
        sigma: float, input noise level
        n_ch: Number of choices. (def: 3, int)
        ob_nch: bool, states whether we add number of choices
        to the observation space. 
        ob_histblock: bool, states whether we add number of current trial 
        history block to the observation space (from TrialHistory wrapper).
    """
    metadata = {
        'description': '''N-alternative forced choice task in which the subject
         has to integrate N stimuli to decide which one is higher
          on average.''',
        'paper_link': 'https://www.nature.com/articles/nn.2123',
        'paper_name': 'Decision-making with multiple alternatives',
        'tags': ['perceptual', 'n-alternative', 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.,
                 stim_scale=1., n_ch=3, ob_nch=False,
                 ob_histblock=False):

        super().__init__(dt=dt)
        self.n = n_ch
        self.choices = np.arange(n_ch)
        self.ob_nch = ob_nch
        self.ob_histblock = ob_histblock

        assert isinstance(n_ch, int), 'n_ch must be integer'
        assert n_ch > 1, 'n_ch must be at least 2'
        assert isinstance(ob_histblock, bool), 'ob_histblock \
                                                must be True/False'
        assert isinstance(ob_nch, bool), 'ob_nch \
                                                must be True/False'
        n_ch_factor = 1.84665761*np.log(n_ch)-0.04102044
        # The strength of evidence, modulated by stim_scale.
        self.cohs = np.array([0, 6.4, 12.8, 25.6, 51.2])*n_ch_factor*stim_scale
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)
        self.timing = {
            'fixation': ('constant', 500),
            'stimulus': ('truncated_exponential', [330, 80, 1500]),
            'decision': ('constant', 500)}
        if timing:
            self.timing.update(timing)

        self.abort = False

        # Action and observation spaces
        self.action_space = spaces.Discrete(n_ch+1)
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(n_ch + ob_nch +
                                                   ob_histblock + 1,),
                                            dtype=np.float32)
        self.ob_dict = {'fixation': 0, 'stimulus': range(1, n_ch + 1)}
        if ob_nch:
            self.ob_dict.update({'Active choices': n_ch + ob_nch})
        if ob_histblock:
            self.ob_dict.update({'Current block': n_ch + ob_nch + ob_histblock})

        self.act_dict = {'fixation': 0, 'choice': range(1, n_ch + 1)}

    def new_trial(self, **kwargs):
        """
        new_trial() is called when a trial ends to generate the next trial.
        The following variables are created:
            ground_truth: correct response for the trial
            coh: stimulus coherence (evidence) for the trial
            ob: observation
        """
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        #  Controling whether ground_truth and/or choices is passed.
        if 'ground_truth' in kwargs.keys():
            ground_truth = kwargs['ground_truth']
        elif 'n_ch' in kwargs.keys():
            ground_truth = self.rng.choice(np.arange(kwargs['n_ch']))
        else:
            ground_truth = self.rng.choice(self.choices)

        self.trial = {
            'ground_truth': ground_truth,
            'coh': self.rng.choice(self.cohs),
        }
        self.trial.update(kwargs)
        self.add_period(['fixation', 'stimulus', 'decision'], after=0,
                        last_period=True)

        self.add_ob(1, 'fixation', where='fixation')
        stim = np.ones(self.n) * (1 - self.trial['coh']/100)/2
        stim[self.trial['ground_truth']] = (1 + self.trial['coh']/100)/2
        self.add_ob(stim, 'stimulus', where='stimulus')

        #  Adding active nch and/or current history block to observations.
        if self.ob_nch:
            if 'n_ch' in kwargs.keys():
                self.add_ob(kwargs['n_ch'], where='Active choices')
            else:
                self.add_ob(len(self.choices), where='Active choices')
        if self.ob_histblock and 'curr_block' in kwargs.keys():
            # We add 1 to make it visible in plots
            self.add_ob(kwargs['curr_block']+1, where='Current block')

        #  Adding noise to stimulus observations
        if 'n_ch' in kwargs.keys():
            self.add_randn(0, self.sigma, 'stimulus',
                           where=np.arange(kwargs['n_ch'])+1)
        else:
            self.add_randn(0, self.sigma, 'stimulus', where='stimulus')
        self.set_groundtruth(self.act_dict['choice'][ground_truth], 'decision')

    def _step(self, action, **kwargs):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """
        # ---------------------------------------------------------------------
        # Reward and observations
        # ---------------------------------------------------------------------
        new_trial = False

        obs = self.ob_now
        gt = self.gt_now

        reward = 0
        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward = self.rewards['correct']
                    self.performance = 1
                else:
                    reward = self.rewards['fail']
        return obs, reward, False, {'new_trial': new_trial, 'gt': gt}


if __name__ == '__main__':
    env = nalt_PerceptualDecisionMaking()
    ngym.utils.plot_env(env)
