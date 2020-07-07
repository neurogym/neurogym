#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ready-set-go task."""

from __future__ import division

import numpy as np
from gym import spaces
import neurogym as ngym


class ReadySetGo(ngym.PeriodEnv):
    r"""Agents have to measure and produce different time intervals.

    Args:
        gain: Controls the measure that the agent has to produce. (def: 1, int)
        prod_margin: controls the interval around the ground truth production
            time within which the agent receives proportional reward
    """
    metadata = {
        'paper_link': 'https://www.sciencedirect.com/science/article/pii/' +
        'S0896627318304185',
        'paper_name': '''Flexible Sensorimotor Computations through Rapid
        Reconfiguration of Cortical Dynamics''',
        'tags': ['timing', 'go-no-go', 'supervised']
    }

    def __init__(self, dt=80, rewards=None, timing=None, gain=1,
                 prod_margin=0.2):
        super().__init__(dt=dt)
        self.prod_margin = prod_margin

        self.gain = gain

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': ('constant', 100),
            'ready': ('constant', 83),
            'measure': ('choice', [800, 1500]),
            'set': ('constant', 83)}
        if timing:
            self.timing.update(timing)

        self.abort = False
        # set action and observation space
        self.action_space = spaces.Discrete(2)  # (fixate, go)
        self.act_dict = {'fixation': 0, 'go': 1}
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)
        self.ob_dict = {'fixation': 0, 'ready': 1, 'set': 2}

    def new_trial(self, **kwargs):
        measure = self.sample_time('measure')
        self.trial = {
            'measure': measure,
            'gain': self.gain
        }
        self.trial.update(kwargs)

        self.trial['production'] = measure * self.trial['gain']

        self.add_period(['fixation', 'ready'])
        self.add_period('measure', duration=measure, after='fixation')
        self.add_period('set', after='measure')
        self.add_period('production', duration=2*self.trial['production'],
                        after='set', last_period=True)

        self.add_ob(1, where='fixation')
        self.set_ob(0, 'production', where='fixation')
        self.add_ob(1, 'ready', where='ready')
        self.add_ob(1, 'set', where='set')

        # set ground truth
        gt = np.zeros((int(2*self.trial['production']/self.dt),))
        gt[int(self.trial['production']/self.dt)] = 1
        self.set_groundtruth(gt, 'production')

    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward and inputs
        # ---------------------------------------------------------------------
        trial = self.trial
        reward = 0
        obs = self.ob_now
        gt = self.gt_now
        new_trial = False
        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']
        if self.in_period('production'):
            if action == 1:
                new_trial = True  # terminate
                # time from end of measure:
                t_prod = self.t - self.end_t['measure']
                eps = abs(t_prod - trial['production'])
                # actual production time
                eps_threshold = self.prod_margin*trial['production']+25
                if eps > eps_threshold:
                    reward = self.rewards['fail']
                else:
                    reward = (1. - eps/eps_threshold)**1.5
                    reward = max(reward, 0.1)
                    reward *= self.rewards['correct']
                    self.performance = 1

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt}


class MotorTiming(ngym.PeriodEnv):
    """Agents have to produce different time intervals
    using different effectors (actions).

    Args:
        prod_margin: controls the interval around the ground truth production
                    time within which the agent receives proportional reward
    """
    #  TODO: different actions not implemented
    metadata = {
        'paper_link': 'https://www.nature.com/articles/s41593-017-0028-6',
        'paper_name': '''Flexible timing by temporal scaling of
         cortical responses''',
        'tags': ['timing', 'go-no-go', 'supervised']
    }

    def __init__(self, dt=80, rewards=None, timing=None, prod_margin=0.2):
        super().__init__(dt=dt)
        self.prod_margin = prod_margin
        self.production_ind = [0, 1]
        self.intervals = [800, 1500]

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': ('constant', 500),  # XXX: not specified
            'cue': ('uniform', [1000, 3000]),
            'set': ('constant', 50)}
        if timing:
            self.timing.update(timing)

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

        self.add_period(['fixation', 'cue', 'set'])
        self.add_period('production', duration=2*self.trial['production'],
                        after='set', last_period=True)

        self.set_ob([1, 0, 0, 0], 'fixation')
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
        self.set_groundtruth(gt, 'production')

    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward and inputs
        # ---------------------------------------------------------------------
        trial = self.trial
        reward = 0
        obs = self.ob_now
        gt = self.gt_now
        new_trial = False
        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']
        if self.in_period('production'):
            if action == 1:
                new_trial = True  # terminate
                t_prod = self.t - self.end_t['set']  # time from end of measure
                eps = abs(t_prod - trial['production'])
                # actual production time
                eps_threshold = self.prod_margin*trial['production']+25
                if eps > eps_threshold:
                    reward = self.rewards['fail']
                else:
                    reward = (1. - eps/eps_threshold)**1.5
                    reward = max(reward, 0.1)
                    reward *= self.rewards['correct']
                    self.performance = 1

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt}


class OneTwoThreeGo(ngym.PeriodEnv):
    r"""Agents reproduce time intervals based on two samples.

    Args:
        prod_margin: controls the interval around the ground truth production
                    time within which the agent receives proportional reward
    """
    metadata = {
        'paper_link': 'https://www.nature.com/articles/s41593-019-0500-6',
        'paper_name': "Internal models of sensorimotor integration "
                      "regulate cortical dynamics",
        'tags': ['timing', 'go-no-go', 'supervised']
    }

    def __init__(self, dt=80, rewards=None, timing=None, prod_margin=0.2):
        super().__init__(dt=dt)

        self.prod_margin = prod_margin

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': ('truncated_exponential', [400, 100, 800]),
            'target': ('truncated_exponential', [1000, 500, 1500]),
            's1': ('constant', 100),
            'interval1': ('choice', [600, 700, 800, 900, 1000]),
            's2': ('constant', 100),
            'interval2': ('constant', 0),
            's3': ('constant', 100),
            'interval3': ('constant', 0),
            'response': ('constant', 1000)}
        if timing:
            self.timing.update(timing)

        self.abort = False
        # set action and observation space
        self.action_space = spaces.Discrete(2)  # (fixate, go)
        self.act_dict = {'fixation': 0, 'go': 1}
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)
        self.ob_dict = {'fixation': 0, 'stimulus': 1, 'target': 2}

    def new_trial(self, **kwargs):
        interval = self.sample_time('interval1')
        self.trial = {
            'interval': interval,
        }
        self.trial.update(kwargs)

        self.add_period(['fixation', 'target', 's1'])
        self.add_period('interval1', duration=interval, after='s1')
        self.add_period('s2', after='interval1')
        self.add_period('interval2', duration=interval, after='s2')
        self.add_period('s3', after='interval2')
        self.add_period('interval3', duration=interval, after='s3')
        self.add_period('response', after='interval3', last_period=True)

        self.add_ob(1, where='fixation')
        self.add_ob(1, ['s1', 's2', 's3'], where='stimulus')
        self.add_ob(1, where='target')
        self.set_ob(0, 'fixation', where='target')

        # set ground truth
        self.set_groundtruth(self.act_dict['go'], period='response')

    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward and inputs
        # ---------------------------------------------------------------------
        trial = self.trial
        reward = 0
        obs = self.ob_now
        gt = self.gt_now
        new_trial = False
        if self.in_period('interval3') or self.in_period('response'):
            if action == 1:
                new_trial = True  # terminate
                # time from end of measure:
                t_prod = self.t - self.end_t['s3']
                eps = abs(t_prod - trial['interval'])  # actual production time
                eps_threshold = self.prod_margin*trial['interval']+25
                if eps > eps_threshold:
                    reward = self.rewards['fail']
                else:
                    reward = (1. - eps/eps_threshold)**1.5
                    reward = max(reward, 0.1)
                    reward *= self.rewards['correct']
                    self.performance = 1
        else:
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt}


if __name__ == '__main__':
    env = ReadySetGo(dt=50)
    ngym.utils.plot_env(env, num_steps=100, def_act=0)

