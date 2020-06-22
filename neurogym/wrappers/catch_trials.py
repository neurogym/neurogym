#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import neurogym as ngym
import numpy as np


class CatchTrials(ngym.TrialWrapper):
    """Catch trials.

    Introduces catch trials in which the reward for a correct choice is
    modified (e.g. is set to the reward for an incorrect choice). Note
    that the wrapper only changes the reward associated to a correct
    answer and does not change the ground truth. Thus, the catch trial
    affect a pure supervised learning setting.

    Args:
        catch_prob: Catch trial probability. (def: 0.1, float)
        stim_th: Percentile of stimulus distribution below which catch trials
            are allowed (experimenter might decide not to have catch
            trials when  stimulus is too obvious). (def: 50, int)
        start: Number of trials after which the catch trials can occur.
            (def: 0, int)
        alt_rew: reward given in catch trials
    """
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

    def __init__(self, env, catch_prob=0.1, stim_th=None, start=0, alt_rew=0):
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
        self.catch_trial = False
        self.alt_rew = alt_rew
        # number of trials after which the prob. of catch trials is != 0
        self.start = start

    def new_trial(self, **kwargs):
        coh = self.task.rng.choice(self.task.cohs)
        if self.stim_th is not None:
            if coh <= self.stim_th:
                self.catch_trial = self.task.rng.rand() < self.catch_prob
            else:
                self.catch_trial = False
        else:
            self.catch_trial = self.task.rng.rand() < self.catch_prob
        kwargs.update({'coh': coh})
        self.env.new_trial(**kwargs)

    def step(self, action, new_tr_fn=None):
        ntr_fn = new_tr_fn or self.new_trial
        obs, reward, done, info = self.env.step(action, new_tr_fn=ntr_fn)
        if info['new_trial']:
            info['catch_trial'] = self.catch_trial
            if self.catch_trial:
                reward = self.alt_rew
        return obs, reward, done, info
