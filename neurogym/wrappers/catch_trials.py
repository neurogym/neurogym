#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:23:36 2019

@author: molano
"""
import neurogym as ngym
import numpy as np


class CatchTrials(ngym.TrialWrapper):
    metadata = {
        'description': 'Introduces catch trials in which the reward for' +
        ' a correct choice is modified (e.g. is set to the reward for an' +
        ' incorrect choice). Note that the wrapper only changes the reward' +
        ' associated to a correct answer and does not change the ground' +
        ' truth. Thus, the catch trial affect a pure supervised learning' +
        ' setting.',
        'paper_link': None,
        'paper_name': None,
    }

    def __init__(self, env, catch_prob=0.1, stim_th=50, start=0):
        """
        Introduces catch trials in which the reward for a correct choice is
        modified (e.g. is set to the reward for an incorrect choice). Note
        that the wrapper only changes the reward associated to a correct
        answer and does not change the ground truth. Thus, the catch trial
        affect a pure supervised learning setting.
        catch_prob: Catch trial probability. (def: 0.1, float)
        stim_th: Percentile of stimulus distribution below which catch trials
        are allowed (in some cases, experimenter might decide not to have catch
        trials when  stimulus is very obvious). (def: 50, int)
        start: Number of trials after which the catch trials can occur.
        (def: 0, int)
        """
        super().__init__(env)
        self.env = env
        # we get the original task, in case we are composing wrappers
        env_aux = env
        while env_aux.__class__.__module__.find('wrapper') != -1:
            env_aux = env.env
        self.task = env_aux
        self.catch_prob = catch_prob
        if stim_th is not None:
            self.stim_th = np.percentile(self.task.cohs, stim_th)
        else:
            self.stim_th = None
        self.r_correct_original = self.task.rewards['correct']
        self.catch_trial = False
        # number of trials after which the prob. of catch trials is != 0
        self.start = start

    def new_trial(self, **kwargs):
        self.task.rewards['correct'] = self.r_correct_original
        coh = self.task.rng.choice(self.task.cohs)
        if self.stim_th is not None:
            if coh <= self.stim_th:
                self.catch_trial = self.task.rng.random() < self.catch_prob
            else:
                self.catch_trial = False
        else:
            self.catch_trial = self.task.rng.random() < self.catch_prob
        if self.catch_trial:
            self.task.rewards['correct'] = self.task.rewards['fail']
        kwargs.update({'coh': coh})
        self.env.new_trial(**kwargs)

    def _step(self, action):
        return self.env._step(action)

    def step(self, action):
        obs, reward, done, info = self._step(action)
        if info['new_trial']:
            if self.task.num_tr > self.start:
                info['catch_trial'] = self.catch_trial
                self.new_trial()
            else:
                info['catch_trial'] = False
        return obs, reward, done, info

    def seed(self, seed=None):
        self.task.seed(seed=seed)
        # keeps track of the repeating prob of the current block
        self.curr_block = self.task.rng.choice([0, 1])
