#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 11:37:41 2019

@author: molano
"""
from gym.envs.registration import register
register(
    id='Mante-v0',
    entry_point='mante:Mante',
    max_episode_steps=100000,
    reward_threshold=90.0,
)

register(
    id='Priors-v0',
    entry_point='priors:Priors',
    max_episode_steps=100000,
    reward_threshold=90.0,
)

register(
    id='DualTask-v0',
    entry_point='dual_task:DualTask',
    max_episode_steps=100000,
    reward_threshold=90.0,
)

register(
    id='Romo-v0',
    entry_point='romo:Romo',
    max_episode_steps=100000,
    reward_threshold=90.0,
)


register(
    id='RDM-v0',
    entry_point='rdm:RDM',
    max_episode_steps=100000,
    reward_threshold=90.0,
)

register(
    id='padoaSch-v0',
    entry_point='padoa_sch:PadoaSch',
    max_episode_steps=100000,
    reward_threshold=90.0,
)

register(
    id='pdWager-v0',
    entry_point='pd_wager:PDWager',
    max_episode_steps=100000,
    reward_threshold=90.0,
)

register(
    id='DPA-v0',
    entry_point='dpa:DPA',
    max_episode_steps=100000,
    reward_threshold=90.0,
)

register(
    id='GNG-v0',
    entry_point='gng:GNG',
    max_episode_steps=100000,
    reward_threshold=90.0,
)

register(
    id='RDM_hist-v0',
    entry_point='rdm_hist:RDM_hist',
    max_episode_steps=100000,
    reward_threshold=90.0,
)

register(
    id='ReadySetGo-v0',
    entry_point='readysetgo:ReadySetGo',
    max_episode_steps=100000,
    reward_threshold=90.0,
)

register(
    id='DelayedMatchSample-v0',
    entry_point='delaymatchsample:DelayedMatchToSample',
    max_episode_steps=100000,
    reward_threshold=90.0,
)

register(
    id='DawTwoStep-v0',
    entry_point='dawtwostep:DawTwoStep',
    max_episode_steps=100000,
    reward_threshold=90.0,
)


def all_envs():
    """
    used in baselines/run.py to check if a task is from neurogym
    """
    return ['rdm', 'dpa', 'mante', 'romo', 'gng', 'padoa_sch', 'pd_wager',
            'rdm_hist', 'readysetgo', 'DelayedMatchSample-v0', 'DawTwoStep-v0']
