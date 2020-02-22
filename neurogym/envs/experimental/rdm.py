#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Random dot motion task."""

import numpy as np
from gym import spaces

import neurogym as ngym
from neurogym.utils import tasktools


class RDM(ngym.PeriodEnv):
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
        self.add_period('fixation', after=0)
        self.add_period('stimulus', after='fixation')
        self.add_period('decision', after='stimulus', last_period=True)
        # ---------------------------------------------------------------------
        # Observations
        # ---------------------------------------------------------------------
        signed_coh = coh if ground_truth == 1 else -coh
        self.add_ob(period='fixation', value=1, where='fixation')
        self.add_ob(period='stimulus', value=(1 + signed_coh / 100) / 2, where='stimulus1')
        self.add_ob(period='stimulus', value=(1 - signed_coh / 100) / 2, where='stimulus2')
        self.add_randn(period='stimulus', sigma=self.sigma_dt)
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
                reward += self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward += self.rewards['correct']
                else:
                    reward += self.rewards['fail']

        return self.obs_now, reward, False, {'new_trial': new_trial, 'gt': gt}



# TODO: Ground truth and action have different space,
# making it difficult for SL and RL to work together
class Reaching1D(ngym.PeriodEnv):
    metadata = {
        'description': 'The agent has to reproduce the angle indicated' +
        ' by the observation.',
        'paper_link': 'https://science.sciencemag.org/content/233/4771/1416',
        'paper_name': 'Neuronal population coding of movement direction',
        'timing': {
            'fixation': ('constant', 500),
            'reach': ('constant', 500)},
        'tags': ['motor', 'steps action space']
    }

    def __init__(self, dt=100, rewards=None, timing=None):
        """
        The agent has to reproduce the angle indicated by the observation.
        dt: Timestep duration. (def: 100 (ms), int)
        rewards:
            R_CORRECT: given when correct. (def: +1., float)
            R_FAIL: given when incorrect. (def: -0.1, float)
        timing: Description and duration of periods forming a trial.
        """
        super().__init__(dt=dt, timing=timing)
        # Rewards
        reward_default = {'R_CORRECT': +1., 'R_FAIL': -0.1}
        if rewards is not None:
            reward_default.update(rewards)
        self.R_CORRECT = reward_default['R_CORRECT']
        self.R_FAIL = reward_default['R_FAIL']

        # action and observation spaces
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(32,),
                                            dtype=np.float32)
        self.ob_dict = {'self': range(16, 32),
                        'target': range(16)}
        self.action_space = spaces.Discrete(3)
        self.act_dict = {'fixation': 0,
                         'left': 1,
                         'right': 2,
                         }
        self.theta = np.arange(0, 2*np.pi, 2*np.pi/16)
        self.state = np.pi

    def new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        self.state = np.pi
        self.trial = {
            'ground_truth': self.rng.uniform(0, np.pi*2)
        }
        self.trial.update(kwargs)
        # ---------------------------------------------------------------------
        # Periods
        # ---------------------------------------------------------------------
        self.add_period('fixation', after=0)
        self.add_period('reach', after='fixation', last_period=True)

        target = np.cos(self.theta - self.trial['ground_truth'])
        self.add_ob('reach', value=target, where='target')

        self.set_groundtruth('fixation', np.pi)
        self.set_groundtruth('reach', self.trial['ground_truth'])
        self.dec_per_dur = (self.end_ind['reach'] - self.start_ind['reach'])

    def _step(self, action):
        ob = self.obs_now
        ob[16:] = np.cos(self.theta - self.state)
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
                np.max((self.R_CORRECT-tasktools.circular_dist(self.state-gt),
                        self.R_FAIL))
            norm_rew = (reward-self.R_FAIL)/(self.R_CORRECT-self.R_FAIL)
            self.performance += norm_rew/self.dec_per_dur

        return ob, reward, False, {'new_trial': False}


if __name__ == '__main__':
    env = Reaching1D(dt=20)
    # env = RDM(dt=20, timing={'stimulus': ('constant', 500)})
    from neurogym.tests.test_envs import test_speed
    test_speed(env)
    ngym.utils.plot_env(env, num_steps_env=100, def_act=1)
