#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 17:16:10 2020

@author: molano

Usage:

    import neurogym as ngym

    import gym
    kwargs = {'dt': 100, 'tr_hist_kwargs': {'probs': 0.9}}
    # Make supervised dataset
    tasks = ngym.get_collection('priors')
    envs = [gym.make(task, **kwargs) for task in tasks]

"""
import gym
import neurogym.wrappers as wrappers


def priors_v0(tr_hist_kwargs={'probs': 0.9}, var_nch_kwargs={}, **task_kwargs):
    env = gym.make('NAltPerceptualDecisionMaking-v0', **task_kwargs)
    print(tr_hist_kwargs)
    env = wrappers.TrialHistoryEvolution(env, **tr_hist_kwargs)
    env = wrappers.Variable_nch(env, **var_nch_kwargs)
    env = wrappers.PassAction(env)
    env = wrappers.PassReward(env)
    return env
