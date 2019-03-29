#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 11:37:41 2019

@author: molano
"""

# TODO: Consider moving this file to __init__ of neurogym to prevent
# the re-register error
from gym.envs.registration import register
import gym
# all_tasks = ['rdm', 'dpa', 'mante', 'romo', 'gng', 'padoa_sch', 'pd_wager',
#              'readysetgo', 'DelayedMatchSample-v0', 'DawTwoStep-v0',
#              'MatchingPenny-v0', 'Bandit-v0']
all_tasks = {'Mante-v0': 'mante:Mante', 'Romo-v0': 'romo:Romo',
             'RDM-v0': 'rdm:RDM', 'padoaSch-v0': 'padoa_sch:PadoaSch',
             'pdWager-v0': 'pd_wager:PDWager', 'DPA-v0': 'dpa:DPA',
             'GNG-v0': 'gng:GNG', 'ReadySetGo-v0': 'readysetgo:ReadySetGo',
             'DelayedMatchSample-v0': 'delaymatchsample:DelayedMatchToSample',
             'DawTwoStep-v0': 'dawtwostep:DawTwoStep',
             'MatchingPenny-v0': 'matchingpenny:MatchingPenny',
             'Bandit-v0': 'bandit:Bandit'}


def register_neuroTask(id_task):
    for env in gym.envs.registry.all():
        if env.id == id_task:
            return
    register(id=id_task, entry_point=all_tasks[id_task])


def all_envs():
    """
    used in baselines/run.py and baselines/common/cmd_util.py
    to check if a task is from neurogym
    """
    return all_tasks.keys()
