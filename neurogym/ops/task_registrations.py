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
all_tasks = {'Mante-v0': 'neurogym.envs.mante:Mante',
             'Romo-v0': 'neurogym.envs.romo:Romo',
             'RDM-v0': 'neurogym.envs.rdm:RDM',
             'RDM-v1': 'neurogym.envs.rdm_v1:RDM',
             'padoaSch-v0': 'neurogym.envs.padoa_sch:PadoaSch',
             'pdWager-v0': 'neurogym.envs.pd_wager:PDWager',
             'DPA-v0': 'neurogym.envs.dpa:DPA',
             'GNG-v0': 'neurogym.envs.gng:GNG',
             'ReadySetGo-v0': 'neurogym.envs.readysetgo:ReadySetGo',
             'DelayedMatchSample-v0':
                 'neurogym.envs.delaymatchsample:DelayedMatchToSample',
             'DawTwoStep-v0': 'neurogym.envs.dawtwostep:DawTwoStep',
             'MatchingPenny-v0': 'neurogym.envs.matchingpenny:MatchingPenny',
             'Bandit-v0': 'neurogym.envs.bandit:Bandit',
             'DelayedResponse-v0': 'neurogym.envs.delayresponse:DR',
             'NAltRDM-v0': 'neurogym.envs.nalt_rdm:nalt_RDM'}


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