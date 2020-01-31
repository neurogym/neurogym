#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:49:35 2019

@author: molano
"""
from gym import spaces
import numpy as np
import itertools
import gym
from neurogym.meta import info
# XXX: implemented without relying on core.trTrialWrapper


class Combine():
    metadata = {
        'description': 'Allows to combine two tasks, one of which working as' +
        ' the distractor task.',
        'paper_link': 'https://www.biorxiv.org/content/10.1101/433409v3',
        'paper_name': '''Response outcomes gate the impact of expectations
         on perceptual decisions''',
        'distractor': 'Distractor task. (no default value)',
        'delay': 'Time when the distractor task appears. (def: 800 (ms))',
        'mix': 'Probabilities for the different trial types' +
        ' (only main, only distractor, both). (def: (.5, .0, .5))',
        'share_action_space': 'Whether the two task share the same action' +
        ' space. (def: True)',
        'defaults': 'Default actions for each task. (def: [0, 0])',
        'trial_cue': 'Whether to show the type of trial as a cue'
    }

    def __init__(self, env, distractor, delay=800,
                 dt=100, mix=(.3, .3, .4), share_action_space=True,
                 defaults=[0, 0], trial_cue=False):
        self.share_action_space = share_action_space
        self.trial_cue = trial_cue
        self.t = 0
        self.dt = dt
        self.delay = delay
        self.delay_on = True
        self.env = env
        self.distractor = distractor
        # default behavior
        self.defaults = defaults
        # sub-tasks probabilities
        self.mix = mix
        num_act1 = self.env.action_space.n
        num_act2 = self.distractor.action_space.n
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
                       shape=(self.env.observation_space.shape[0] +
                              self.distractor.observation_space.shape[0] +
                              1*self.trial_cue, ),
                       dtype=np.float32)
        # start trials
        self.env_on = True
        self.distractor_on = True
        self.new_trial()
        self.reward_range = (np.min([self.env.reward_range[0],
                                     self.distractor.reward_range[0]]),
                             np.max([self.env.reward_range[1],
                                     self.distractor.reward_range[1]]))
        self.metadata = self.env.metadata
        self.metadata.update(self.distractor.metadata)

    def new_trial(self):
        # decide type of trial
        self.task_type = self.env.rng.choices([0, 1, 2], weights=self.mix)[0]
        if self.task_type == 0:
            self.env_on = True
            self.distractor_on = False
            self.tmax = self.env.tmax
        elif self.task_type == 1:
            self.env_on = False
            self.distractor_on = True
            self.tmax = self.distractor.tmax
        else:
            self.env_on = True
            self.distractor_on = True
            self.tmax = self.env.tmax + self.distractor.tmax

    def reset(self):
        obs_env = self.env.reset()
        obs_distractor = self.distractor.reset()
        obs = np.concatenate((obs_env, obs_distractor), axis=0)
        if self.trial_cue:
            cue = np.array([self.task_type])
            obs = np.concatenate((cue, obs), axis=0)
        return obs

    def _step(self, action):
        info = {}
        action1, action2 = self.action_split[action]
        # get outputs from main task
        if self.env_on:
            obs1, reward1, done1, info1 = self.env.step(action1)
            new_trial1 = info1['new_trial']
            self.env_on = not new_trial1
            info['env_info'] = info1
        else:
            obs1, reward1, done1, info1 = self.standby_step(self.env)
            info['env_info'] = None
        # get outputs from distractor task
        if self.t > self.delay and self.distractor_on:
            obs2, reward2, done2, info2 = self.distractor.step(action2)
            new_trial2 = info2['new_trial']
            self.distractor_on = not new_trial2
            info['distractor_info'] = info2
        else:
            obs2, reward2, done2, info2 = self.standby_step(self.distractor)
            info['distractor_info'] = None
        # new trial?
        if not self.env_on and not self.distractor_on:
            info['new_trial'] = True
            info['config'] = [self.env_on, self.distractor_on]
        else:
            info['new_trial'] = False

        # build joint observation
        obs = np.concatenate((obs1, obs2), axis=0)
        if self.trial_cue:
            cue = np.array([self.task_type])
            obs = np.concatenate((cue, obs), axis=0)
        done = done1  # done whenever the task 1 is done

        # ground truth
        if self.share_action_space:
            if info1['gt'] == info2['gt']:
                info['gt'] = info1['gt']  # task 1 is the default task
                reward = reward1 + reward2
            elif (info1['gt'] != self.defaults[0] and
                  info2['gt'] == self.defaults[1]):
                info['gt'] = info1['gt']
                reward = reward1
            elif (info1['gt'] == self.defaults[0] and
                  info2['gt'] != self.defaults[1]):
                info['gt'] = info2['gt']
                reward = reward2
        else:
            info['gt'] = np.concatenate((info2['gt'], info2['gt']))
        return obs, reward, done, info

    def step(self, action):
        obs, reward, done, info = self._step(action)
        self.t += self.dt  # increment within trial time count
        if info['new_trial']:
            self.t = 0
            self.new_trial()

        return obs, reward, done, info

    def standby_step(self, env):
        """
        creates fake outputs when env is not active
        """
        obs = np.zeros((env.observation_space.shape[0], ))
        rew = 0
        done = False
        gt = 0
        info = {'new_trial': False, 'gt': gt}
        return obs, rew, done, info


if __name__ == '__main__':
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
    env = Combine(env, distractor, delay=300, mix=(.3, .3, .4),
                  share_action_space=True, defaults=[0, 0],
                  trial_cue=True)
    info.plot_struct(env, num_steps_env=100, num_steps_plt=100)
