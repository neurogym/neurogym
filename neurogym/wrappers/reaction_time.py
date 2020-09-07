#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 15:07:21 2019

@author: molano
"""
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

    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def reset(self, step_fn=None):
        if step_fn is None:
            step_fn = self.step
        return self.env.reset(step_fn=step_fn)

    def step(self, action):
        assert 'stimulus' in self.env.start_t.keys(),\
            'Reaction time wrapper requires a stimulus period'
        assert 'decision' in self.env.start_t.keys(),\
            'Reaction time wrapper requires a decision period'
        if self.t_ind == 0:
            self.env.start_t['decision'] = self.env.start_t['stimulus']
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info
