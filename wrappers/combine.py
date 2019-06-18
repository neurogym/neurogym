#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:49:35 2019

@author: molano
"""

from gym import spaces
import numpy as np
import itertools


class combine():
    """
    combines two different tasks
    """
    def __init__(self, env1, env2, delay=800, dt=100, mix=[.3, .3, .4],
                 share_action_space=True, defaults=[0, 0]):
        self.share_action_space = share_action_space
        self.t = 0
        self.dt = dt
        self.delay = delay
        self.delay_on = True
        self.env1 = env1
        self.env2 = env2
        # default behavior
        self.defaults = defaults
        # sub-tasks probabilities
        self.mix = mix
        num_act1 = self.env1.action_space.n
        num_act2 = self.env2.action_space.n
        if self.share_action_space:
            assert num_act1 == num_act2, "action spaces must be of same size"
            self.num_act = num_act1
            self.action_space = spaces.Discrete(self.num_act)
            act_list = np.arange(self.num_act).reshape((self.num_act, 1))
            self.action_split = np.hstack((act_list, act_list))
        else:
            self.num_act = self.num_act1 * self.num_act2
            self.action_space = spaces.Discrete(self.num_act)
            self.action_split =\
                list(itertools.product(np.arange(self.num_act1),
                                       np.arange(self.num_act2)))

        self.observation_space = \
            spaces.Box(-np.inf, np.inf,
                       shape=(self.env1.observation_space.shape[0] +
                              self.env2.observation_space.shape[0], ),
                       dtype=np.float32)
        # start trials
        self.env1_on = True
        self.env2_on = True
        self._new_trial()
        self.reward_range = (np.min([self.env1.reward_range[0],
                                     self.env2.reward_range[0]]),
                             np.max([self.env1.reward_range[1],
                                     self.env2.reward_range[1]]))
        self.metadata = self.env1.metadata
        self.metadata.update(self.env2.metadata)
        self.spec = None

    def _new_trial(self):
        task_type = np.random.choice([0, 1, 2], p=self.mix)
        if task_type == 0:
            self.env1.trial = self.env1._new_trial()
            self.env1_on = True
            self.env2_on = False
        elif task_type == 1:
            self.env2.trial = self.env2._new_trial()
            self.env1_on = False
            self.env2_on = True
        else:
            self.env1.trial = self.env1._new_trial()
            self.env2.trial = self.env2._new_trial()
            self.env1_on = True
            self.env2_on = True

    def reset(self):
        return np.concatenate((self.env1.reset(), self.env2.reset()), axis=0)

    def step(self, action):
        action1, action2 = self.action_split[action]
        if self.env1_on:
            obs1, reward1, done1, info1 = self.env1._step(action1)
            self.env1_on = not info1['new_trial']
            new_trial1 = info1['new_trial']
        else:
            obs1, reward1, done1, info1 = self.standby_step(1)
            new_trial1 = False

        if self.t > self.delay and self.env2_on:
            obs2, reward2, done2, info2 = self.env2._step(action2)
            self.env2_on = not info2['new_trial']
            new_trial2 = info2['new_trial']
        else:
            obs2, reward2, done2, info2 = self.standby_step(2)
            new_trial2 = False

        if not self.env1_on and not self.env2_on:
            self.t = 0
            self._new_trial()

        self.t += self.dt

        obs = np.concatenate((obs1, obs2), axis=0)
        reward = reward1 + reward2
        done = done1  # done whenever the task 1 is done

        # new trial information
        info = {}
        # ground truth
        if self.share_action_space:
            if (info1['gt'] == info2['gt']).all:
                info = {'gt': info1['gt']}  # task 1 is the default task
            elif (info1['gt'][self.defaults[0]] == 0 and
                  info2['gt'][self.defaults[1]] == 1):
                info = {'gt': info1['gt']}
            elif (info1['gt'][self.defaults[0]] == 1 and
                  info2['gt'][self.defaults[1]] == 0):
                info = {'gt': info2['gt']}
        if new_trial1 or new_trial2:
            info = {'new_trial': True}
        else:
            info = {'new_trial': False}

        return obs, reward, done, info

    def standby_step(self, env):
        if env == 1:
            obs = np.zeros((self.env1.observation_space.shape[0], ))
        else:
            obs = np.zeros((self.env2.observation_space.shape[0], ))
        rew = 0
        done = False
        info = {'new_trial': False, 'gt': np.zeros((self.num_act,))}
        return obs, rew, done, info

    def seed(self, seed=None):  # seeding with task 1
        return self.env1.seed()
