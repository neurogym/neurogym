#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:22:19 2019

@author: linux
"""
import gym


def build_env(env_id, args):
    all_args = vars(args)
    env_keys = {'dt'}
    env_args = {x: all_args[x] for x in env_keys}
    env = gym.make(env_id, **env_args)
    if all_args['trial_hist']:
        env = trial_hist.TrialHistory(env)
    env = manage_data.manage_data(env)