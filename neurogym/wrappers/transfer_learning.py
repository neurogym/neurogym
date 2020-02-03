#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 10:54:49 2020

@author: manuel
"""
from gym import spaces
import numpy as np
import itertools
import gym
# XXX: implemented without relying on core.trTrialWrapper


class TransferLearning():
    metadata = {
        'description': 'Allows training on several tasks sequencially.',
        'paper_link': '',
        'paper_name': '',
        'envs': 'List with environments. (list)',
        'num_tr': 'Number of trials to train on each task. (list)',
        'share_action_space': 'Whether the tasks share the same action' +
        ' space. (def: True)',
        'task_cue': 'Whether to show the current task as a cue. (def:False)'
    }

    def __init__(self, envs, num_tr_per_task, dt=100, share_action_space=True,
                 task_cue=False):
        self.share_action_space = share_action_space
        self.t = 0
        self.dt = dt
        self.envs = envs
        self.num_tr_per_task = num_tr_per_task
        self.task_cue = task_cue
        # sub-tasks probabilities
        num_act = self.envs[0].action_space.n
        if self.share_action_space:
            for ind_env in range(1, len(self.envs)):
                na = self.envs[ind_env].action_space.n
                assert num_act == na, "action spaces must be of same size"
            self.num_act = num_act
            self.action_space = spaces.Discrete(self.num_act)
            act_list = np.arange(self.num_act).reshape((self.num_act, 1))
            self.action_split = np.hstack((act_list, act_list))
        else:
            for ind_env in range(1, len(self.envs)):
                na = self.envs[ind_env].action_space.n
                num_act *= na
            self.num_act = num_act
            self.action_space = spaces.Discrete(self.num_act)
            self.action_split =\
                list(itertools.product(np.arange(self.num_act1),
                                       np.arange(self.num_act2)))
        obs_shape = np.sum([x.observation_space.shape[0] for x in envs])
        self.observation_space = \
            spaces.Box(-np.inf, np.inf,
                       shape=(obs_shape + 1*self.trial_cue,),
                       dtype=np.float32)
        # start trials
        self.env_counter = 0
        self.tr_counter = 0
        self.current_env = self.envs[self.env_counter]
        self.metadata = self.envs[0].metadata
        for ind_env in range(1, len(self.envs)):
            self.metadata.update(self.envs[ind_env].metadata)

    def new_trial(self):
        # decide type of trial
        if self.tr_counter > self.num_tr_per_task[self.env_counter]:
            self.env_counter += 1
            self.current_env = self.envs[self.env_counter]
            self.tr_counter = 0
            self.reset()
        self.tr_counter += 1

    def reset(self):
        return self.current_env.reset()

    def step(self, action):
        obs, reward, done, info = self.current_env.step(action)
        if self.trial_cue:
            cue = np.array([self.task_c])
            obs = np.concatenate((cue, obs), axis=0)

        return obs, reward, done, info


if __name__ == '__main__':
    import neurogym as ngym
    #    task = 'DelayPairedAssociation-v0'
    #    KWARGS = {'dt': 100, 'timing': {'fixation': ('constant', 0),
    #                                    'stim1': ('constant', 100),
    #                                    'delay_btw_stim': ('constant', 500),
    #                                    'stim2': ('constant', 100),
    #                                    'delay_aft_stim': ('constant', 100),
    #                                    'decision': ('constant', 200)}}
    #    env = gym.make(task, **KWARGS)
    task = 'GoNogo-v0'
    KWARGS = {'dt': 100, 'timing': {'fixation': ('constant', 0),
                                    'stimulus': ('constant', 100),
                                    'resp_delay': ('constant', 1200),
                                    'decision': ('constant', 100)}}
    env = gym.make(task, **KWARGS)

    task = 'GoNogo-v0'
    KWARGS = {'dt': 100, 'timing': {'fixation': ('constant', 0),
                                    'stimulus': ('constant', 100),
                                    'resp_delay': ('constant', 100),
                                    'decision': ('constant', 100)}}
    distractor = gym.make(task, **KWARGS)
    env = TransferLearning(env, share_action_space=False,
                           trial_cue=True)
    ngym.utils.plot_env(env, num_steps_env=100, num_steps_plt=100)
