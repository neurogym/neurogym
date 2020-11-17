#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This is try to code spatial suppression motion task


import numpy as np
from gym import spaces
import neurogym as ngym
import sys


class SpatialSuppressMotion(ngym.TrialEnv):
    '''
    Spatial suppression motion task. This task is useful to study center-surround interaction in monkey MT and human psychophysical performance in motion perception.

    Tha task is derived from (Tadin et al. Nature, 2003). In this task, there is no fixation or decision stage. We only present a stimulus and a subject needs to perform a 4-AFC motion direction judgement. The ground-truth is the probabilities for choosing the four directions at a given time point. The probabilities depend on stimulus contrast and size, and the probabilities are derived from emprically measured human psychophysical performance.

    In this version, the input size is 4 (directions) x 8 (size) = 32 neurons. This setting aims to simulate four pools (8 neurons in each pool) of neurons that are selective for four directions. 

    Args:
        <dt>: millisecs per image frame, default: 8.3 (given 120HZ monitor)
        <win_size>: size per image frame
        <timing>: millisecs, stimulus duration, default: 8.3 * 36 frames ~ 300 ms. 
            This is the longest duration we need (i.e., probability reach ceilling)
    
    Note that please input default seq_len = 36 frames when creating dataset object.


    '''
    metadata = {
        'paper_link': 'https://www.nature.com/articles/nature01800',
        'paper_name': '''Perceptual consequences of centre–surround antagonism in visual motion processing ''',
        'tags': ['perceptual', 'plaid', 'motion', 'center-surround']
    }

    def __init__(self, dt=8.3, timing={'stimulus':300}, rewards=None):
        super().__init__(dt=dt)
        
        from numpy import pi
        
        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        # Timing
        self.timing = {
            'stimulus': 300,  # we only need stimulus period for psychophysical task
            }
        if timing:
            self.timing.update(timing)
            
        self.abort = False

        # define action space four directions 
        self.action_space = spaces.Box(0, 1, shape=(4,), dtype=np.float32) # the probabilities for four direction
        
        # define observation space
        self.observation_space = spaces.Box(
            0, np.inf, shape=(32,), dtype=np.float32) # observation space, 4 directions * 8 sizes
        # larger stimulus could elicit more neurons to fire

        self.directions = [1, 2, 3, 4] # motion direction left/right/up/down
        self.theta = [-pi/2, pi/2, 0, pi] # direction angle of the four directions
        self.directions_anti = [2, 1, 4, 3]
        self.directions_ortho = [[3, 4], [3, 4], [1, 2], [1, 2]]


    def _new_trial(self, diameter=None, contrast=None, direction=None):
        '''
        To define a stimulus, we need diameter, contrast, duration, direction
        <diameter>: 0~8, stimulus size in norm units
        <contrast>: 0~1, stimulus contrast
        <direction>: int(1/2/3/4), left/right/up/down
        '''
        #import ipdb;ipdb.set_trace();import matplotlib.pyplot as plt;
        
        # if no stimulus information provided, we random sample stimulus parameters  
        if direction is None:
            direction = self.rng.choice(self.directions)
        if contrast is None:
            #contrast = self.rng.uniform(0, 1) # stimlus contrast
            contrast = self.rng.choice([0.05, 0.99])  # Low contrast, and high contrast
        # here we only consider small-low contrast and large-high contrast condition
        if contrast == 0.05: # small low contrast
            diameter = 1
        elif contrast == 0.99: # large high contrast
            diameter = 8
        
        trial = {
            'diameter': diameter,
            'contrast': contrast,
            'direction': direction,
        }

        # Periods and Timing
        # we only need stimulus period for this psychophysical task
        periods = ['stimulus']
        self.add_period(periods)

        # We need ground_truth
        # the probablities to choose four directions given stimulus parameters
        trial['ground_truth'] = self.getgroundtruth(trial)
        
        # create the stimulus
        ob = self.view_ob(period='stimulus')
        ob = np.zeros((ob.shape[0], ob.shape[1]))        
        
        stim = (np.cos(np.array(self.theta) - self.theta[direction-1])+1) * contrast / 2       
        stim = np.tile(stim, [diameter, 1])

        if diameter != 8:
            tmp = np.zeros((8-diameter, 4))
            stim = np.vstack((stim, tmp)).T.flatten()
        stim = stim.T.flatten()
        stim = np.tile(stim, [ob.shape[0], 1])
        
        ob = stim.copy()

        # set observation and groundtruth
        self.set_ob(ob, 'stimulus')
        self.set_groundtruth(trial['ground_truth'], 'stimulus')

        return trial

    def _step(self, action):
        '''
        We need output for every single step until to the end, no need to check action every step and calculate reward. Just let this function complete all steps.
        
        The _step function is useful for making a choice early in a trial or the situation when breaking the fixation.
 
        '''
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now
        # # observations
        # if self.in_period('stimulus'): # start a new trial once step into decision stage       
        #          new_trial = True
        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}
    
  
    def getgroundtruth(self, trial):
        '''
        The utility function to obtain ground truth probabilities for four direction

        Input trial is a dict, contains fields <duration>, <contrast>, <diameter>

        We output a (4,) tuple indicate the probabilities to perceive left/right/up/down direction. This label comes from emprically measured human performance 
        '''
        from scipy.interpolate import interp1d
        from numpy import zeros
        
        #duration = [5, 7.296, 10.65, 15.54, 22.67, 33.08, 48.27, 70.44, 102.8]
        #frame_ind = [self.envelope(i/1000)[1] for i in duration]
        frame_ind = [8, 9, 10, 13, 15, 18, 21, 28, 36, 37, 38, 39]
        xx = [1, 2, 3, 4, 5, 6, 7]
        yy = [0.249] * 7

        frame_ind = xx + frame_ind  # to fill in the first a few frames
        frame_ind = [i-1 for i in frame_ind] # frame index start from
        
        seq_len = self.view_ob(period='stimulus').shape[0]
        xnew = np.arange(seq_len)

        if trial['contrast'] > 0.5:
            # large size (11 deg radius), High contrast
            prob_corr = yy + [0.249, 0.249, 0.249, 0.27, 0.32, 0.4583, 0.65, 0.85, 0.99, 0.99, 0.99, 0.99]
            prob_anti = yy + [0.249, 0.29, 0.31, 0.4, 0.475, 0.4167, 0.3083, 0.075, 0.04, 0.04, 0.03, 0.03]
        
        elif trial['contrast'] < 0.5:
            # large size (11 deg radius), low contrast
            prob_corr = yy + [0.25, 0.26, 0.2583, 0.325, 0.45, 0.575, 0.875, 0.933, 0.99, 0.99, 0.99,0.99]
            prob_anti = yy + [0.25, 0.26, 0.2583, 0.267, 0.1417, 0.1167, 0.058, 0.016, 0.003, 0.003, 0.003, 0.003]
        
        corr_prob = interp1d(frame_ind, prob_corr, kind='slinear', fill_value='extrapolate')(xnew)
        anti_prob = interp1d(frame_ind, prob_anti,
                             kind='slinear', fill_value='extrapolate')(xnew)
        ortho_prob = (1-(corr_prob + anti_prob)) / 2
        
        direction = trial['direction']-1
        direction_anti = self.directions_anti[direction]-1
        direction_ortho = [i-1 for i in self.directions_ortho[direction]]
        
        gt = zeros((4, seq_len))
        gt[direction, :] = corr_prob
        gt[direction_anti, :] = anti_prob
        gt[direction_ortho, :] = ortho_prob

        #import ipdb;ipdb.set_trace();import matplotlib.pyplot as plt;
        gt = gt.T 
        # gt is a seq_len x 4 numpy array
        return gt
