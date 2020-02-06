#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ready-set-go task."""

from __future__ import division

import numpy as np
from gym import spaces
import neurogym as ngym


class ReadySetGo(ngym.PeriodEnv):
    metadata = {
        'description': '''Agents have to measure and produce different time
         intervals.''',
        'paper_link': 'https://www.sciencedirect.com/science/article/pii/' +
        'S0896627318304185',
        'paper_name': '''Flexible Sensorimotor Computations through Rapid
        Reconfiguration of Cortical Dynamics''',
        'timing': {
            'fixation': ('constant', 100),
            'ready': ('constant', 83),
            'measure': ('choice', [800, 1500]),
            'set': ('constant', 83)},
        'tags': ['timing', 'go-no-go', 'supervised']
    }

    def __init__(self, dt=80, rewards=None, timing=None, gain=1):
        """
        Agents have to measure and produce different time intervals.
        dt: Timestep duration. (def: 80 (ms), int)
        rewards:
            R_ABORTED: given when breaking fixation. (def: -0.1, float)
            R_CORRECT: given when correct. (def: +1., float)
            R_FAIL: given when incorrect. (def: 0., float)
        timing: Description and duration of periods forming a trial.
        gain: Controls the measure that the agent has to produce. (def: 1, int)
        """
        super().__init__(dt=dt, timing=timing)

        self.gain = gain

        # Rewards
        reward_default = {'R_ABORTED': -0.1, 'R_CORRECT': +1.,
                          'R_FAIL': 0.}
        if rewards is not None:
            reward_default.update(rewards)
        self.R_ABORTED = reward_default['R_ABORTED']
        self.R_CORRECT = reward_default['R_CORRECT']
        self.R_FAIL = reward_default['R_FAIL']

        self.abort = False
        # set action and observation space
        self.action_space = spaces.Discrete(2)  # (fixate, go)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)

    def new_trial(self, **kwargs):
        measure = (self.timing_fn['measure']() // self.dt) * self.dt
        self.trial = {
            'measure': measure,
            'gain': self.gain
        }
        self.trial.update(kwargs)

        self.trial['production'] = measure * self.trial['gain']

        self.add_period('fixation', after=0)
        self.add_period('ready', after='fixation')
        self.add_period('measure', duration=measure, after='fixation')
        self.add_period('set', after='measure')
        self.add_period('production', duration=2*self.trial['production'],
                        after='set', last_period=True)

        self.set_ob('fixation', [1, 0, 0])
        self.set_ob('ready', [0, 1, 0])
        self.set_ob('set', [0, 0, 1])
        # set ground truth
        gt = np.zeros((int(2*self.trial['production']/self.dt),))
        gt[int(self.trial['production']/self.dt)] = 1
        self.set_groundtruth('production', gt)

    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward and inputs
        # ---------------------------------------------------------------------
        trial = self.trial
        reward = 0
        obs = self.obs_now
        gt = self.gt_now
        new_trial = False
        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.R_ABORTED
        if self.in_period('production'):
            if action == 1:
                new_trial = True  # terminate
                # time from end of measure:
                t_prod = self.t - self.end_t['measure']
                eps = abs(t_prod - trial['production'])
                # actual production time
                eps_threshold = 0.2*trial['production']+25
                if eps > eps_threshold:
                    reward = self.R_FAIL
                else:
                    reward = (1. - eps/eps_threshold)**1.5
                    reward = min(reward, 0.1)
                    reward *= self.R_CORRECT

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt}


class MotorTiming(ngym.PeriodEnv):
    #  TODO: different actions not implemented
    metadata = {
        'description': 'Agents have to produce different time' +
        ' intervals using different effectors (actions).' +
        ' [different actions not implemented]',
        'paper_link': 'https://www.nature.com/articles/s41593-017-0028-6',
        'paper_name': '''Flexible timing by temporal scaling of
         cortical responses''',
        'timing': {
            'fixation': ('constant', 500),  # XXX: not specified
            'cue': ('uniform', [1000, 3000]),
            'set': ('constant', 50)},
        'tags': ['timing', 'go-no-go', 'supervised']
    }

    def __init__(self, dt=80, rewards=None, timing=None):
        """
        Agents have to produce different time intervals
        using different effectors (actions).
        dt: Timestep duration. (def: 80 (ms), int)
        rewards:
            R_ABORTED: given when breaking fixation. (def: -0.1, float)
            R_CORRECT: given when correct. (def: +1., float)
            R_FAIL: given when incorrect. (def: 0., float)
        timing: Description and duration of periods forming a trial.
        """
        super().__init__(dt=dt, timing=timing)

        self.production_ind = [0, 1]
        self.intervals = [800, 1500]

        # Rewards
        reward_default = {'R_ABORTED': -0.1, 'R_CORRECT': +1.,
                          'R_FAIL': 0.}
        if rewards is not None:
            reward_default.update(rewards)
        self.R_ABORTED = reward_default['R_ABORTED']
        self.R_CORRECT = reward_default['R_CORRECT']
        self.R_FAIL = reward_default['R_FAIL']

        self.abort = False
        # set action and observation space
        self.action_space = spaces.Discrete(2)  # (fixate, go)
        # Fixation, Interval indicator x2, Set
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4,),
                                            dtype=np.float32)

    def new_trial(self, **kwargs):
        self.trial = {
            'production_ind': self.rng.choice(self.production_ind)
        }
        self.trial.update(kwargs)

        self.trial['production'] = self.intervals[self.trial['production_ind']]

        self.add_period('fixation', after=0)
        self.add_period('cue', after='fixation')
        self.add_period('set', after='cue')
        self.add_period('production', duration=2*self.trial['production'],
                        after='set', last_period=True)

        self.set_ob('fixation', [1, 0, 0, 0])
        ob = self.view_ob('cue')
        ob[:, 0] = 1
        ob[:, self.trial['production_ind']+1] = 1
        ob = self.view_ob('set')
        ob[:, 0] = 1
        ob[:, self.trial['production_ind'] + 1] = 1
        ob[:, 3] = 1
        # set ground truth
        gt = np.zeros((int(2*self.trial['production']/self.dt),))
        gt[int(self.trial['production']/self.dt)] = 1
        self.set_groundtruth('production', gt)

    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward and inputs
        # ---------------------------------------------------------------------
        trial = self.trial
        reward = 0
        obs = self.obs_now
        gt = self.gt_now
        new_trial = False
        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.R_ABORTED
        if self.in_period('production'):
            if action == 1:
                new_trial = True  # terminate
                t_prod = self.t - self.end_t['set']  # time from end of measure
                eps = abs(t_prod - trial['production'])
                # actual production time
                eps_threshold = 0.2*trial['production']+25
                if eps > eps_threshold:
                    reward = self.R_FAIL
                else:
                    reward = (1. - eps/eps_threshold)**1.5
                    reward = min(reward, 0.1)
                    reward *= self.R_CORRECT

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt}


if __name__ == '__main__':
    env = MotorTiming()
    ngym.utils.plot_env(env, num_steps_env=100, def_act=0)
#    env = ReadySetGo()
#    ngym.utils.plot_env(env, num_steps_env=100, def_act=0)
