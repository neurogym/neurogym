# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 18:00:32 2019

@author: MOLANO
"""

"""
A parametric working memory task, based on

  Neuronal population coding of parametric working memory.
  O. Barak, M. Tsodyks, & R. Romo, JNS 2010.

  http://dx.doi.org/10.1523/JNEUROSCI.1875-10.2010

"""
from __future__ import division

import numpy as np

from pyrl import tasktools

# Inputs
inputs = tasktools.to_map('FIXATION', 'F-POS', 'F-NEG')

# Actions
actions = tasktools.to_map('FIXATE', '>', '<')

# Trial conditions
gt_lts = ['>', '<']
fpairs = [(18, 10), (22, 14), (26, 18), (30, 22), (34, 26)]
n_conditions = len(gt_lts) * len(fpairs)

# Training
n_gradient   = n_conditions
n_validation = 20*n_conditions

# Slow down the learning
lr          = 0.002
baseline_lr = 0.002

# Input noise
sigma = np.sqrt(2*100*0.001)

# Epoch durations
fixation  = 750
f1        = 500
delay_min = 3000 - 300
delay_max = 3000 + 300
f2        = 500
decision  = 500
tmax      = fixation + f1 + delay_max + f2 + decision

def get_condition(rng, dt, context={}):
    #-------------------------------------------------------------------------------------
    # Epochs
    #-------------------------------------------------------------------------------------

    delay = context.get('delay')
    if delay is None:
        delay = tasktools.uniform(rng, dt, delay_min, delay_max)

    durations = {
        'fixation':   (0, fixation),
        'f1':         (fixation, fixation + f1),
        'delay':      (fixation + f1, fixation + f1 + delay),
        'f2':         (fixation + f1 + delay, fixation + f1 + delay + f2),
        'decision':   (fixation + f1 + delay + f2, tmax),
        'tmax':       tmax
        }
    time, epochs = tasktools.get_epochs_idx(dt, durations)

    gt_lt = context.get('gt_lt')
    if gt_lt is None:
        gt_lt = tasktools.choice(rng, gt_lts)

    fpair = context.get('fpair')
    if fpair is None:
        fpair = tasktools.choice(rng, fpairs)

    return {
        'durations': durations,
        'time':      time,
        'epochs':    epochs,
        'gt_lt':     gt_lt,
        'fpair':     fpair
        }

# Rewards
R_ABORTED = -1
R_CORRECT = +1

# Input scaling
fall = np.ravel(fpairs)
fmin = np.min(fall)
fmax = np.max(fall)

def scale(f):
    return (f - fmin)/(fmax - fmin)

def scale_p(f):
    return (1 + scale(f))/2

def scale_n(f):
    return (1 - scale(f))/2

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
            status['choice']   = None
            reward = R_ABORTED
    elif t-1 in epochs['decision']:
        if a == actions['>']:
            status['continue'] = False
            status['choice']   = '>'
            status['correct']  = (trial['gt_lt'] == '>')
            if status['correct']:
                reward = R_CORRECT
        elif a == actions['<']:
            status['continue'] = False
            status['choice']   = '<'
            status['correct']  = (trial['gt_lt'] == '<')
            if status['correct']:
                reward = R_CORRECT

    #-------------------------------------------------------------------------------------
    # Inputs
    #-------------------------------------------------------------------------------------

    if trial['gt_lt'] == '>':
        f1, f2 = trial['fpair']
    else:
        f2, f1 = trial['fpair']

    u = np.zeros(len(inputs))
    if t not in epochs['decision']:
        u[inputs['FIXATION']] = 1
    if t in epochs['f1']:
        u[inputs['F-POS']] = scale_p(f1) + rng.normal(scale=sigma)/np.sqrt(dt)
        u[inputs['F-NEG']] = scale_n(f1) + rng.normal(scale=sigma)/np.sqrt(dt)
    if t in epochs['f2']:
        u[inputs['F-POS']] = scale_p(f2) + rng.normal(scale=sigma)/np.sqrt(dt)
        u[inputs['F-NEG']] = scale_n(f2) + rng.normal(scale=sigma)/np.sqrt(dt)

    #-------------------------------------------------------------------------------------

    return u, reward, status

def terminate(perf):
    p_decision, p_correct = tasktools.correct_2AFC(perf)

    return p_decision >= 0.99 and p_correct >= 0.97