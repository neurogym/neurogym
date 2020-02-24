#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Random dot motion task."""

import numpy as np
from gym import spaces

import neurogym as ngym


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
        'tags': ['perceptual', 'two-alternative', 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None, stimEv=1.):
        """
        Two-alternative forced choice task in which the subject has to
        integrate two stimuli to decide which one is higher on average.

        Parameters:
        dt: Timestep duration. (def: 100 (ms), int)
        rewards:
            R_ABORTED: given when breaking fixation. (def: -0.1, float)
            R_CORRECT: given when correct. (def: +1., float)
            R_FAIL: given when incorrect. (def: 0., float)
        timing: Description and duration of periods forming a trial.
        stimEv: Controls the difficulty of the experiment. (def: 1., float)
        """
        super().__init__(dt=dt, timing=timing)
        self.choices = [1, 2]  # [left, right]
        # cohs specifies the amount of evidence (which is modulated by stimEv)
        self.cohs = np.array([0, 6.4, 12.8, 25.6, 51.2]) * stimEv
        # Input noise
        sigma = np.sqrt(2 * 100 * 0.01)
        self.sigma_dt = sigma / np.sqrt(self.dt)

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.abort = False

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)
        self.ob_dict = {'fixation': 0,
                        'stimulus1': 1,
                        'stimulus2': 2}

        self.action_space = spaces.Discrete(3)
        self.act_dict = {'fixation': 0,
                         'choice1': 1,
                         'choice2': 2}

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
        self.add_period(['fixation', 'stimulus', 'decision'], after=0, last_period=True)
        # ---------------------------------------------------------------------
        # Observations
        # ---------------------------------------------------------------------
        signed_coh = coh if ground_truth == 1 else -coh
        self.add_ob(1, period='fixation', where='fixation')
        self.add_ob((1 + signed_coh / 100) / 2, period='stimulus', where='stimulus1')
        self.add_ob((1 - signed_coh / 100) / 2, period='stimulus', where='stimulus2')
        self.add_randn(0, self.sigma_dt, 'stimulus')
        # ---------------------------------------------------------------------
        # Ground truth
        # ---------------------------------------------------------------------
        self.set_groundtruth(ground_truth, 'decision')

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
                reward += self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward += self.rewards['correct']
                else:
                    reward += self.rewards['fail']

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

    def __init__(self, dt=100, rewards=None, timing=None, stimEv=1.):
        """
        Agents have to integrate two stimuli and report which one is
        larger on average after a delay.
        dt: Timestep duration. (def: 100 (ms), int)
        rewards:
            R_ABORTED: given when breaking fixation. (def: -0.1, float)
            R_CORRECT: given when correct. (def: +1., float)
            R_FAIL: given when incorrect. (def: -1., float)
        timing: Description and duration of periods forming a trial.
        stimEv: Controls the difficulty of the experiment. (def: 1., float)
        """
        super().__init__(dt=dt, timing=timing)
        self.choices = [1, 2]
        # cohs specifies the amount of evidence (which is modulated by stimEv)
        self.cohs = np.array([0, 6.4, 12.8, 25.6, 51.2])*stimEv
        # Input noise
        sigma = np.sqrt(2*100*0.01)
        self.sigma_dt = sigma / np.sqrt(self.dt)

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1.,
                          'fail': 0.}
        if rewards:
            self.rewards.update(rewards)
        self.rewards['abort'] = self.rewards['abort']
        self.rewards['correct'] = self.rewards['correct']
        self.rewards['fail'] = self.rewards['fail']

        self.abort = False

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
        self.set_ob([1, 0, 0], 'fixation')
        stim = self.view_ob('stimulus')
        stim[:, 0] = 1
        stim[:, 1:] = (1 - self.trial['coh']/100)/2
        stim[:, self.trial['ground_truth']] = (1 + self.trial['coh']/100)/2
        stim[:, 1:] +=\
            self.rng.randn(stim.shape[0], 2) * self.trial['sigma_dt']

        self.set_ob([1, 0, 0], 'delay')

        self.set_groundtruth(self.trial['ground_truth'], 'decision')

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

        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']
        elif self.in_period('decision') and action != 0:
            new_trial = True
            if action == gt:
                reward = self.rewards['correct']
                self.performance = 1
            elif action == 3 - gt:  # 3-action is the other act
                reward = self.rewards['fail']

        info = {'new_trial': new_trial, 'gt': gt}
        return self.obs_now, reward, False, info


if __name__ == '__main__':
    env = PerceptualDecisionMaking(dt=20,
                                   timing={'stimulus': ('constant', 500)})
    from neurogym.tests.test_envs import test_speed
    test_speed(env)
    ngym.utils.plot_env(env, num_steps_env=100, def_act=1)
    # env = PerceptualDecisionMakingDelayResponse()
    # ngym.utils.plot_env(env, num_steps_env=100, def_act=1)
