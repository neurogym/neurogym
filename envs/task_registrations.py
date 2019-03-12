#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 11:37:41 2019

@author: molano
"""

# TODO: Consider moving this file to __init__ of neurogym to prevent the re-register error
from gym.envs.registration import register
register(
    id='Mante-v0',
    entry_point='mante:Mante',
    reward_threshold=90.0,
)

register(
    id='Priors-v0',
    entry_point='priors:Priors',
    reward_threshold=90.0,
)

register(
    id='DualTask-v0',
    entry_point='dual_task:DualTask',
    reward_threshold=90.0,
)

register(
    id='Romo-v0',
    entry_point='romo:Romo',
    reward_threshold=90.0,
)


register(
    id='RDM-v0',
    entry_point='rdm:RDM',
    reward_threshold=90.0,
)

register(
    id='padoaSch-v0',
    entry_point='padoa_sch:PadoaSch',
    reward_threshold=90.0,
)

register(
    id='pdWager-v0',
    entry_point='pd_wager:PDWager',
    reward_threshold=90.0,
)

register(
    id='DPA-v0',
    entry_point='dpa:DPA',
    reward_threshold=90.0,
)

register(
    id='GNG-v0',
    entry_point='gng:GNG',
    reward_threshold=90.0,
)

register(
    id='RDM_hist-v0',
    entry_point='rdm_hist:RDM_hist',
    reward_threshold=90.0,
)

register(
    id='ReadySetGo-v0',
    entry_point='readysetgo:ReadySetGo',
    reward_threshold=90.0,
)

register(
    id='DelayedMatchSample-v0',
    entry_point='delaymatchsample:DelayedMatchToSample',
    reward_threshold=90.0,
)

register(
    id='DawTwoStep-v0',
    entry_point='dawtwostep:DawTwoStep',
    reward_threshold=90.0,
)

register(
    id='MatchingPenny-v0',
    entry_point='matchingpenny:MatchingPenny',
    reward_threshold=90.0,
)


def all_envs():
    """
    used in baselines/run.py to check if a task is from neurogym
    """
    return ['rdm', 'dpa', 'mante', 'romo', 'gng', 'padoa_sch', 'pd_wager',
            'rdm_hist', 'readysetgo', 'DelayedMatchSample-v0', 'DawTwoStep-v0',
            'MatchingPenny-v0']
