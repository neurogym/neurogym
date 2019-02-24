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
