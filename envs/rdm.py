#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 13:48:19 2019

@author: molano


Perceptual decision-making task, based on

  Bounded integration in parietal cortex underlies decisions even when viewing
  duration is dictated by the environment.
  R Kiani, TD Hanks, & MN Shadlen, JNS 2008.

  http://dx.doi.org/10.1523/JNEUROSCI.4761-07.2008

"""
from __future__ import division

import numpy as np

from pyrl import tasktools

# Time step
dt = 1

# Inputs
inputs = tasktools.to_map('FIXATION', 'LEFT', 'RIGHT')

# Actions
actions = tasktools.to_map('FIXATE', 'CHOOSE-LEFT', 'CHOOSE-RIGHT')

# Trial conditions
left_rights  = [-1, 1]
cohs         = [0, 6.4, 12.8, 25.6, 51.2]
n_conditions = len(left_rights)*len(cohs)

# Training
n_gradient   = n_conditions
n_validation = 100*n_conditions

# Input noise
sigma = np.sqrt(2*100*0.01)

# Durations
fixation      = 750
stimulus_min  = 80
stimulus_mean = 330
stimulus_max  = 1500
decision      = 500
tmax          = fixation + stimulus_max + decision

# Rewards
R_ABORTED = -1
R_CORRECT = +1

def get_condition(rng, dt, context={}):
    #-------------------------------------------------------------------------------------
    # Epochs
    #-------------------------------------------------------------------------------------

    stimulus = context.get('stimulus')
    if stimulus is None:
        stimulus = tasktools.truncated_exponential(rng, dt, stimulus_mean,
                                                   xmin=stimulus_min, xmax=stimulus_max)

    durations = {
        'fixation':  (0, fixation),
        'stimulus':  (fixation, fixation + stimulus),
        'decision':  (fixation + stimulus, fixation + stimulus + decision),
        'tmax':      tmax
        }
    time, epochs = tasktools.get_epochs_idx(dt, durations)

    #-------------------------------------------------------------------------------------
    # Trial
    #-------------------------------------------------------------------------------------

    left_right = context.get('left_right')
    if left_right is None:
        left_right = rng.choice(left_rights)

    coh = context.get('coh')
    if coh is None:
        coh = rng.choice(cohs)

    return {
        'durations':   durations,
        'time':        time,
        'epochs':      epochs,
        'left_right':  left_right,
        'coh':         coh
        }

# Input scaling
def scale(coh):
    return (1 + coh/100)/2

def get_step(rng, dt, trial, t, a):
    #-------------------------------------------------------------------------------------
    # Reward
    #-------------------------------------------------------------------------------------

    epochs = trial['epochs']
    status = {'continue': True}
    reward = 0
    if t-1 not in epochs['decision']:
        if a != actions['FIXATE']:
            status['continue'] = False
            reward = R_ABORTED
    elif t-1 in epochs['decision']:
        if a == actions['CHOOSE-LEFT']:
            status['continue'] = False
            status['choice']   = 'L'
            status['t_choice'] = t-1
            status['correct']  = (trial['left_right'] < 0)
            if status['correct']:
                reward = R_CORRECT
        elif a == actions['CHOOSE-RIGHT']:
            status['continue'] = False
            status['choice']   = 'R'
            status['t_choice'] = t-1
            status['correct']  = (trial['left_right'] > 0)
            if status['correct']:
                reward = R_CORRECT

    #-------------------------------------------------------------------------------------
    # Inputs
    #-------------------------------------------------------------------------------------

    if trial['left_right'] < 0:
        high = inputs['LEFT']
        low  = inputs['RIGHT']
    else:
        high = inputs['RIGHT']
        low  = inputs['LEFT']

    u = np.zeros(len(inputs))
    if t in epochs['fixation'] or t in epochs['stimulus']:
        u[inputs['FIXATION']] = 1
    if t in epochs['stimulus']:
        u[high] = scale(+trial['coh']) + rng.normal(scale=sigma)/np.sqrt(dt)
        u[low]  = scale(-trial['coh']) + rng.normal(scale=sigma)/np.sqrt(dt)

    #-------------------------------------------------------------------------------------

    return u, reward, status

def terminate(perf):
    p_decision, p_correct = tasktools.correct_2AFC(perf)

    return p_decision >= 0.99 and p_correct >= 0.8
