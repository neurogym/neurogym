#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gym import spaces
import numpy as np
import gym
# XXX: implemented without relying on core.trTrialWrapper


class TransferLearning():
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
        self.t = 0
        self.envs = envs
        self.num_tr_per_task = num_tr_per_task
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
        self.current_env = self.envs[self.env_counter]
        self.metadata = self.envs[0].metadata
        for ind_env in range(1, len(self.envs)):
            self.metadata.update(self.envs[ind_env].metadata)

    def new_trial(self):
        # decide type of trial
        if self.tr_counter >= self.num_tr_per_task[self.env_counter]:
            self.env_counter += 1
            self.current_env = self.envs[self.env_counter]
            self.tr_counter = 1
            self.reset()
            if self.env_counter == len(self.num_tr_per_task):
                self.final_task = True
        self.tr_counter += 1

    def reset(self):
        obs = self.current_env.reset()
        obs = self.modify_obs(obs)

        return obs

    def step(self, action):
        obs, reward, done, info = self.current_env.step(action)
        if info['new_trial'] and not self.final_task:
            self.new_trial()
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
                                    'resp_delay': ('constant', 100),
                                    'decision': ('constant', 100)}}
    env1 = gym.make(task, **KWARGS)

#    task = 'DelayPairedAssociation-v0'
#    KWARGS = {'dt': 100, 'timing': {'fixation': ('constant', 0),
#                                    'stim1': ('constant', 100),
#                                    'delay_btw_stim': ('constant', 500),
#                                    'stim2': ('constant', 100),
#                                    'delay_aft_stim': ('constant', 100),
#                                    'decision': ('constant', 200)}}
#    env2 = gym.make(task, **KWARGS)
#    task = 'Reaching1D-v0'
#    KWARGS = {'dt': 100, 'timing': {'fixation': ('constant', 500),
#                                    'reach': ('constant', 500)}}
#    env2 = gym.make(task, **KWARGS)

    task = 'Detection-v0'
    KWARGS = {'dt': 100, 'timing': {'fixation': ('constant', 100),
                                    'stimulus': ('constant', 500)}}
    env2 = gym.make(task, **KWARGS)

    env = TransferLearning([env1, env2], num_tr_per_task=[2],
                           task_cue=True)
    ngym.utils.plot_env(env, num_steps=60)
