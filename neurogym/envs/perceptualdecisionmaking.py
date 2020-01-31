#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Random dot motion task."""

import numpy as np
from gym import spaces

import neurogym as ngym
from neurogym.meta import info


class PerceptualDecisionMaking(ngym.PeriodEnv):
    metadata = {
        'description': '''Random dot motion task. Two-alternative forced
         choice task in which the subject has to integrate two stimuli to
         decide which one is higher on average.''',
        'paper_link': 'https://www.jneurosci.org/content/12/12/4745',
        'paper_name': '''The analysis of visual motion: a comparison of
        neuronal and psychophysical performance''',
        'timing': {
            'fixation': ('constant', 100),  # TODO: depends on subject
            'stimulus': ('constant', 2000),
            'decision': ('constant', 100)},  # XXX: not specified
        'stimEv': 'Controls the difficulty of the experiment. (def: 1.)',
        'tags': ['perceptual', '2-alternative', 'supervised']
    }

    def __init__(self, dt=100, timing=None, stimEv=1.):
        super().__init__(dt=dt, timing=timing)
        self.choices = [1, 2]  # [left, right]
        # cohs specifies the amount of evidence (which is modulated by stimEv)
        self.cohs = np.array([0, 6.4, 12.8, 25.6, 51.2]) * stimEv
        # Input noise
        sigma = np.sqrt(2 * 100 * 0.01)
        self.sigma_dt = sigma / np.sqrt(self.dt)

        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_FAIL = 0.
        self.abort = False
        # action and observation spaces
        self.action_space = spaces.Discrete(3)
        # observation space: [fixation cue, left stim, right stim]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)

    def new_trial(self, **kwargs):
        """
        new_trial() is called when a trial ends to generate the next trial.
        The following variables are created:
            durations, which stores the duration of the different periods (in
            the case of perceptualDecisionMaking: fixation, stimulus and
            decision periods)
            ground truth: correct response for the trial
            coh: stimulus coherence (evidence) for the trial
            obs: observation
        """
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        self.trial = {
            'ground_truth': self.rng.choice(self.choices),
            'coh': self.rng.choice(self.cohs),
        }
        self.trial.update(kwargs)
        coh = self.trial['coh']
        ground_truth = self.trial['ground_truth']
        # ---------------------------------------------------------------------
        # Periods
        # ---------------------------------------------------------------------
        self.add_period('fixation', after=0)
        self.add_period('stimulus', after='fixation')
        self.add_period('decision', after='stimulus', last_period=True)
        # ---------------------------------------------------------------------
        # Observations
        # ---------------------------------------------------------------------
        # all observation values are 0 by default
        # FIXATION: setting fixation cue to 1 during fixation period
        self.set_ob('fixation', [1, 0, 0])
        # STIMULUS
        stimulus = self.view_ob('stimulus')  # stimulus shape = time x obs-dim
        # setting coherences
        stimulus[:, 1:] = (1 - coh / 100) / 2
        stimulus[:, ground_truth] = (1 + coh / 100) / 2  # coh for correct side
        # adding gaussian noise to stimulus with std = self.sigma_dt
        stimulus[:, 1:] +=\
            np.random.randn(stimulus.shape[0], 2) * self.sigma_dt
        # ---------------------------------------------------------------------
        # Ground truth
        # ---------------------------------------------------------------------
        self.set_groundtruth('decision', ground_truth)

    def _step(self, action):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now
        # observations
        if self.in_period('fixation'):
            if action != 0:  # action = 0 means fixating
                new_trial = self.abort
                reward = self.R_ABORTED
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward = self.R_CORRECT
                else:
                    reward = self.R_FAIL

        return self.obs_now, reward, False, {'new_trial': new_trial, 'gt': gt}


#  TODO: there should be a timeout of 1000ms for incorrect trials
class PerceptualDecisionMakingDelayResponse(ngym.PeriodEnv):
    metadata = {
        'description': 'Agents have to integrate two stimuli and report' +
        ' which one is larger on average after a delay.',
        'paper_link': 'https://www.nature.com/articles/s41586-019-0919-7',
        'paper_name': 'Discrete attractor dynamics underlies persistent' +
        ' activity in the frontal cortex',
        'timing': {
            'fixation': ('constant', 0),
            'stimulus': ('constant', 1150),
            #  TODO: sampling of delays follows exponential
            'delay': ('choice', [300, 500, 700, 900, 1200, 2000, 3200, 4000]),
            # 'go_cue': ('constant', 100), # TODO: Not implemented
            'decision': ('constant', 1500)},
        'stimEv': 'Controls the difficulty of the experiment. (def: 1.)',
        'tags': ['perceptual', 'delayed response', 'two-alternative',
                 'supervised']
    }

    def __init__(self, dt=100, timing=None, stimEv=1.):
        super().__init__(dt=dt, timing=timing)
        self.choices = [1, 2]
        # cohs specifies the amount of evidence (which is modulated by stimEv)
        self.cohs = np.array([0, 6.4, 12.8, 25.6, 51.2])*stimEv
        # Input noise
        sigma = np.sqrt(2*100*0.01)
        self.sigma_dt = sigma / np.sqrt(self.dt)

        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_FAIL = -1.
        self.abort = False
        self.firstcounts = True
        self.first_flag = False
        # action and observation spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)

    def new_trial(self, **kwargs):
        """
        new_trial() is called when a trial ends to generate the next trial.
        The following variables are created:
            durations, which stores the duration of the different periods (in
            the case of perceptualDecisionMaking: fixation, stimulus and
            decision periods)
            ground truth: correct response for the trial
            coh: stimulus coherence (evidence) for the trial
            obs: observation
        """
        self.first_flag = False

        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        self.trial = {
            'ground_truth': self.rng.choice(self.choices),
            'coh': self.rng.choice(self.cohs),
            'sigma_dt': self.sigma_dt,
        }
        self.trial.update(kwargs)

        # ---------------------------------------------------------------------
        # Periods
        # ---------------------------------------------------------------------
        self.add_period('fixation', after=0)
        self.add_period('stimulus', after='fixation')
        self.add_period('delay', after='stimulus')
        self.add_period('decision', after='delay', last_period=True)

        # define observations
        self.set_ob('fixation', [1, 0, 0])
        stim = self.view_ob('stimulus')
        stim[:, 0] = 1
        stim[:, 1:] = (1 - self.trial['coh']/100)/2
        stim[:, self.trial['ground_truth']] = (1 + self.trial['coh']/100)/2
        stim[:, 1:] +=\
            np.random.randn(stim.shape[0], 2) * self.trial['sigma_dt']

        self.set_ob('delay', [1, 0, 0])

        self.set_groundtruth('decision', self.trial['ground_truth'])

    def _step(self, action):
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
        # rewards
        reward = 0
        # observations
        gt = self.gt_now

        first_trial = np.nan
        if self.in_period('fixation') or self.in_period('delay'):
            if action != 0:
                new_trial = self.abort
                reward = self.R_ABORTED
        elif self.in_period('decision'):
            if action == gt:
                reward = self.R_CORRECT
                new_trial = True
                if ~self.first_flag:
                    first_trial = True
                    self.first_flag = True
            elif action == 3 - gt:  # 3-action is the other act
                reward = self.R_FAIL
                new_trial = self.firstcounts
                if ~self.first_flag:
                    first_trial = False
                    self.first_flag = True

        info = {'new_trial': new_trial,
                'gt': gt,
                'first_trial': first_trial}
        return self.obs_now, reward, False, info


if __name__ == '__main__':
    env = PerceptualDecisionMaking()
    info.plot_struct(env, num_steps_env=50000)
