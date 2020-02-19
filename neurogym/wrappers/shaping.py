#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:38:53 2020

@author: martafradera
"""

import gym
import neurogym as ngym
import numpy as np


class Shaping(ngym.TrialWrapper):
    metadata = {
        'description': '',
        'paper_link': None,
        'paper_name': None,
    }

    def __init__(self, env, init_ph=0, max_num_reps=3):
        """
        """
        super().__init__(env)
        self.env = env
        self.curr_ph = init_ph
        self.counter = 0
        self.action = 0
        self.prev_act = 0
        self.max_num_reps = max_num_reps
        self.first_choice = True
        self.performance = 0
        self.short = False

    def count(self, action):
        '''
        check the last three answers during stage 0 so the network has to
        alternate between left and right
        '''
        if action != 0:
            if action == self.prev_act:
                self.counter += 1
            else:
                self.counter = 1
                self.prev_act = action

    def new_trial(self, **kwargs):
        self.set_phase()
        self.first_choice_rew = None
        if self.curr_ph > 2:
            self.env.performance = 0
            self.env.t = self.env.t_ind = 0
            self.env.num_tr += 1
            if not self.short:
                self.short = True
                periods = self.env.timing.itemss()
                for key, val in periods[:-1]:
                    dist, args = val
                    self.env.timing[key] =\
                        np.max(self.dt,
                               int(self.env.timing[k]/self.short_min))

            self.first_choice = True       
        elif self.curr_ph == 2:
            # first answer counts
            # wrong answer is penalized
            self.durs.update({'delay': (0)})
            self.trial.update({'coh': 100})
            self.trial.update({'sigma_dt': 0})
            self.R_FAIL = -1
            self.firstcounts = True
        elif self.curr_ph == 3:
            # delay component is introduced
            self.trial.update({'coh': 100})
            self.trial.update({'sigma_dt': 0})
        # phase 4: ambiguity component is introduced
        self.env.new_trial()

    def step(self, action):

        if self.curr_ph < 2:
            obs, reward, done, info = self.env._step(action)
            self.env.t += self.env.dt  # increment within trial time count
            self.env.t_ind += 1
            if info['new_trial']:
                if self.curr_ph == 0:
                    # reward when action != fixate
                    # agent cannot go max_num_reps times in a row to same side
                    self.count(action)
                    if self.counter > self.max_num_reps:
                        reward = 0
                        self.performance = 0
                    else:
                        reward = self.env.R_CORRECT
                        self.performance = 1
                elif self.curr_ph == 1:
                    if not self.env.performance:
                        reward = 0
                        info['new_trial'] = False
                    if self.first_choice:
                        self.performance = self.env.performance
                        self.first_choice = False

            if self.env.t > self.env.tmax - self.env.dt and\
               not info['new_trial']:
                info['new_trial'] = True
                self.performance = self.env.performance
                reward += self.r_tmax

            if info['new_trial']:
                info['performance'] = self.performance
                self.new_trial()
        else:
            obs, reward, done, info = self.env.step(action)
            if self.curr_ph == 2:
                reward = max(reward, 0)

        return obs, reward, done, info


if __name__ == '__main__':
    import neurogym as ngym

    task = 'DelayedMatchSample-v0'
    env = gym.make(task)
    env = Shaping(env, init_ph=1)
    ngym.utils.plot_env(env, num_steps_env=100)
