#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Random dot motion task."""

import numpy as np
from gym import spaces

import neurogym as ngym


class PerceptualDecisionMaking(ngym.PeriodEnv):
    """Two-alternative forced choice task in which the subject has to
    integrate two stimuli to decide which one is higher on average.

    Args:
        stim_scale: Controls the difficulty of the experiment. (def: 1., float)
        sigma: float, input noise level
        dim_ring: int, dimension of ring input and output
    """
    metadata = {
        'paper_link': 'https://www.jneurosci.org/content/12/12/4745',
        'paper_name': '''The analysis of visual motion: a comparison of
        neuronal and psychophysical performance''',
        'tags': ['perceptual', 'two-alternative', 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None, stim_scale=1.,
                 sigma=1.5, dim_ring=2):
        super().__init__(dt=dt)
        # The strength of evidence, modulated by stim_scale
        self.cohs = np.array([0, 6.4, 12.8, 25.6, 51.2]) * stim_scale
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': ('constant', 100),  # TODO: depends on subject
            'stimulus': ('constant', 2000),
            'decision': ('constant', 100)}  # XXX: not specified
        if timing:
            self.timing.update(timing)

        self.abort = False

        self.theta = np.linspace(0, 2*np.pi, dim_ring+1)[:-1]
        self.choices = np.arange(dim_ring)

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1+dim_ring,), dtype=np.float32)
        self.ob_dict = {'fixation': 0, 'stimulus': range(1, dim_ring+1)}
        self.action_space = spaces.Discrete(1+dim_ring)
        self.act_dict = {'fixation': 0, 'choice': range(1, dim_ring+1)}

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
        # Trial info
        self.trial = {
            'ground_truth': self.rng.choice(self.choices),
            'coh': self.rng.choice(self.cohs),
        }
        self.trial.update(kwargs)

        coh = self.trial['coh']
        ground_truth = self.trial['ground_truth']
        stim_theta = self.theta[ground_truth]

        # Periods
        self.add_period(['fixation', 'stimulus', 'decision'], after=0,
                        last_period=True)

        # Observations
        self.add_ob(1, period=['fixation', 'stimulus'], where='fixation')
        stim = np.cos(self.theta - stim_theta) * (coh/200) + 0.5
        self.add_ob(stim, 'stimulus', where='stimulus')
        self.add_randn(0, self.sigma, 'stimulus', where='stimulus')

        # Ground truth
        self.set_groundtruth(self.act_dict['choice'][ground_truth], 'decision')

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
                    self.performance = 1
                else:
                    reward += self.rewards['fail']

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}


#  TODO: there should be a timeout of 1000ms for incorrect trials
class PerceptualDecisionMakingDelayResponse(ngym.PeriodEnv):
    """Perceptual decision-making with delayed responses.

    Agents have to integrate two stimuli and report which one is
    larger on average after a delay.

    Args:
        stim_scale: Controls the difficulty of the experiment. (def: 1., float)
    """
    metadata = {
        'paper_link': 'https://www.nature.com/articles/s41586-019-0919-7',
        'paper_name': 'Discrete attractor dynamics underlies persistent' +
        ' activity in the frontal cortex',
        'tags': ['perceptual', 'delayed response', 'two-alternative',
                 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None, stim_scale=1.,
                 sigma=1.5):
        super().__init__(dt=dt)
        self.choices = [1, 2]
        # cohs specifies the amount of evidence (modulated by stim_scale)
        self.cohs = np.array([0, 6.4, 12.8, 25.6, 51.2])*stim_scale
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1.,
                        'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': ('constant', 0),
            'stimulus': ('constant', 1150),
            #  TODO: sampling of delays follows exponential
            'delay': ('choice', [300, 500, 700, 900, 1200, 2000, 3200, 4000]),
            # 'go_cue': ('constant', 100), # TODO: Not implemented
            'decision': ('constant', 1500)}
        if timing:
            self.timing.update(timing)

        self.abort = False

        # action and observation spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)

    def new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        self.trial = {
            'ground_truth': self.rng.choice(self.choices),
            'coh': self.rng.choice(self.cohs),
            'sigma': self.sigma,
        }
        self.trial.update(kwargs)

        # ---------------------------------------------------------------------
        # Periods
        # ---------------------------------------------------------------------
        periods = ['fixation', 'stimulus', 'delay', 'decision']
        self.add_period(periods, after=0, last_period=True)

        # define observations
        self.set_ob([1, 0, 0], 'fixation')
        stim = self.view_ob('stimulus')
        stim[:, 0] = 1
        stim[:, 1:] = (1 - self.trial['coh']/100)/2
        stim[:, self.trial['ground_truth']] = (1 + self.trial['coh']/100)/2
        stim[:, 1:] +=\
            self.rng.randn(stim.shape[0], 2) * self.trial['sigma']

        self.set_ob([1, 0, 0], 'delay')

        self.set_groundtruth(self.trial['ground_truth'], 'decision')

    def _step(self, action):
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
        return self.ob_now, reward, False, info


class PulseDecisionMaking(ngym.PeriodEnv):
    """Pulse-based decision making task.

    Discrete stimuli are presented briefly as pulses.

    Args:
        p_pulse: array-like, probability of pulses for each choice
        n_bin: int, number of stimulus bins
    """
    metadata = {
        'paper_link': 'https://elifesciences.org/articles/11308',
        'paper_name': '''Sources of noise during accumulation of evidence in
        unrestrained and voluntarily head-restrained rats''',
        'tags': ['perceptual', 'two-alternative', 'supervised']
    }

    def __init__(self, dt=10, rewards=None, timing=None, p_pulse=(0.3, 0.7),
                 n_bin=6):
        super().__init__(dt=dt)
        self.p_pulse = p_pulse
        self.n_bin = n_bin

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': ('constant', 500),
            'decision': ('constant', 500)}
        for i in range(n_bin):
            self.timing['cue' + str(i)] = ('constant', 10)
            self.timing['bin' + str(i)] = ('constant', 240)
        if timing:
            self.timing.update(timing)

        self.abort = False

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(3,), dtype=np.float32)
        self.ob_dict = {'fixation': 0, 'stimulus': [1, 2]}
        self.action_space = spaces.Discrete(3)
        self.act_dict = {'fixation': 0, 'choice': [1, 2]}

    def new_trial(self, **kwargs):
        # Trial info
        p1, p2 = self.p_pulse
        if self.rng.random() < 0.5:
            p1, p2 = p2, p1
        pulse1 = (self.rng.random(self.n_bin) < p1) * 1.0
        pulse2 = (self.rng.random(self.n_bin) < p2) * 1.0
        self.trial = {'pulse1': pulse1, 'pulse2': pulse2}
        self.trial.update(kwargs)

        n_pulse1 = sum(pulse1)
        n_pulse2 = sum(pulse2) + self.rng.uniform(-0.1, 0.1)
        ground_truth = int(n_pulse1 < n_pulse2)
        self.trial['ground_truth'] = ground_truth

        # Periods
        periods = ['fixation']
        for i in range(self.n_bin):
            periods += ['cue' + str(i), 'bin' + str(i)]
        periods += ['decision']
        self.add_period(periods, after=0, last_period=True)

        # Observations
        self.add_ob(1, where='fixation')
        for i in range(self.n_bin):
            self.add_ob(pulse1[i], 'cue' + str(i), where=1)
            self.add_ob(pulse2[i], 'cue' + str(i), where=2)
        self.set_ob(0, 'decision')

        # Ground truth
        self.set_groundtruth(self.act_dict['choice'][ground_truth], 'decision')

    def _step(self, action):
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now
        # observations
        if self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward += self.rewards['correct']
                    self.performance = 1
                else:
                    reward += self.rewards['fail']
        else:
            if action != 0:  # action = 0 means fixating
                new_trial = self.abort
                reward += self.rewards['abort']

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}


if __name__ == '__main__':
    env = PerceptualDecisionMaking(dt=20,
                                   timing={'stimulus': ('constant', 500)})
    ngym.utils.plot_env(env, num_steps=100, def_act=1)
    # env = PerceptualDecisionMakingDelayResponse()
    # ngym.utils.plot_env(env, num_steps=100, def_act=1)
