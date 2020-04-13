#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gym import spaces
import numpy as np
import gym
import neurogym as ngym
# XXX: implemented without relying on core.trTrialWrapper


class TransferLearning(ngym.TrialWrapper):
    """Allows training on several tasks sequencially.

    Args:
        envs: List with environments. (list)
        num_tr_per_task: Number of trials to train on each task. (list)
        task_cue: Whether to show the current task as a cue. (def: False, bool)
    """
    metadata = {
        'description': 'Allows training on several tasks sequencially.',
        'paper_link': '',
        'paper_name': ''
        }

    def __init__(self, envs, num_tr_per_task, task_cue=False):
        super().__init__(envs[0])
        self.t = 0
        self.envs = envs
        self.num_tr_per_task = num_tr_per_task
        self.num_tr_per_task.append(10**9)
        self.task_cue = task_cue
        self.final_task = False
        # sub-tasks probabilities
        num_act = 0
        rew_min = np.inf
        rew_max = -np.inf
        for ind_env in range(1, len(self.envs)):
            na = self.envs[ind_env].action_space.n
            num_act = max(num_act, na)
            rew_min = min(rew_min, self.envs[ind_env].reward_range[0])
            rew_max = max(rew_max, self.envs[ind_env].reward_range[1])
        self.num_act = num_act
        self.action_space = spaces.Discrete(self.num_act)

        self.ob_sh = np.max([x.observation_space.shape[0] for x in envs]) +\
            1*self.task_cue
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(self.ob_sh,),
                                            dtype=np.float32)
        # reward range
        self.reward_range = (rew_min, rew_max)
        # start trials
        self.env_counter = 0
        self.tr_counter = 1
        self.env = self.envs[self.env_counter]
        self.metadata = self.envs[0].metadata
        for ind_env in range(1, len(self.envs)):
            self.metadata.update(self.envs[ind_env].metadata)

    def new_trial(self, **kwargs):
        # decide type of trial
        task_done = self.tr_counter >= self.num_tr_per_task[self.env_counter]
        final_task = self.env_counter == len(self.num_tr_per_task)
        if task_done and not final_task:
            self.env_counter += 1
            self.env = self.envs[self.env_counter]
            self.tr_counter = 1
            self.reset()
        self.tr_counter += 1
        self.env.new_trial(**kwargs)

    def step(self, action, new_tr_fn=None):
        ntr_fn = new_tr_fn or self.new_trial
        obs, reward, done, info = self.env.step(action, new_tr_fn=ntr_fn)
        obs = self.modify_obs(obs)
        info['task'] = self.env_counter
        return obs, reward, done, info

    def modify_obs(self, obs):
        extra_sh = self.ob_sh-obs.shape[0]-1*self.task_cue
        obs = np.concatenate((obs, np.zeros((extra_sh,))),
                             axis=0)
        if self.task_cue:
            cue = np.array([self.env_counter])
            obs = np.concatenate((cue, obs), axis=0)
        return obs
