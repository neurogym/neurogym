"""
Noise wrapper.

Created on Thu Feb 28 15:07:21 2019

@author: molano
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import gym


class ReactionTime(gym.Wrapper):  # TODO: Make this a trial wrapper instead?
    """Allow reaction time response.

    Modifies a given environment by allowing the network to act at
    any time after the fixation period.
    """

    metadata = {
        'description': 'Modifies a given environment by allowing the network' +
        ' to act at any time after the fixation period.',
        'paper_link': None,
        'paper_name': None,
    }

    def __init__(self, env, urgency=0.0):
        super().__init__(env)
        self.env = env
        self.urgency = urgency
        self.tr_dur = 0

    def reset(self, step_fn=None):
        if step_fn is None:
            step_fn = self.step
        return self.env.reset(step_fn=step_fn)

    def step(self, action):
        dec = 'decision'
        stim = 'stimulus'
        assert stim in self.env.start_t.keys(),\
            'Reaction time wrapper requires a stimulus period'
        assert dec in self.env.start_t.keys(),\
            'Reaction time wrapper requires a decision period'
        if self.env.t_ind == 0:
            # set start of decision period
            self.env.start_t[dec] = self.env.start_t[stim]+self.env.dt
            # change ground truth accordingly
            self.env.gt[self.start_ind[stim]+1: self.env.end_ind[stim]] =\
                self.env.gt[self.start_ind[dec]]
        obs, reward, done, info = self.env.step(action)
        if info['new_trial']:
            info['tr_dur'] = self.tr_dur
            obs *= 0
        else:
            self.tr_dur = self.env.t_ind
        reward += self.urgency
        return obs, reward, done, info
