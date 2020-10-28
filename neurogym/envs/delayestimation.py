#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""delay-estimation visual working memory task."""

import numpy as np
from gym import spaces

import neurogym as ngym


class delayestimation(ngym.TrialEnv):
    '''
    By Ru-Yuan Zhang (ruyuanzhang@gmail.com)
    Delay-estimation visual working memory task. A subject is asked to memorize the colors (or orientations) of a set of color squares (or bars). After a delay, the subject report the color of a cued object on color wheel (or adjust the orientation of a cued bar). 

    Both orientation and color can be converted to [0, pi) circular variable range.

    This task is similar to anglereproduction task but differs in 1) allowing multiple input objects, and 2) provide continous reward value based on response error


    Args:
        <dt>: delta time
        <sigma>: float, input noise level
        <dim_ring>: int (default:180), dimension of ring input and output
    '''

    metadata = {
        'paper_link': 'https://www.nature.com/articles/nature06860',
        'paper_name': '''Discrete fixed-resolution representations in visual working memory''',
        'tags': ['visual working memory', 'perceptual', 'supervised']
    }

    def __init__(self, dt=50, sigma=1.0, dim_ring=180, timing=None):
        super().__init__(dt=dt)

        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'noresp': -0.1}  # need to change here

        self.timing = {
            'stimulus': 100,
            'delay': 900,
            'decision': 2000
            }
        if timing:
            self.timing.update(timing)

        self.abort = False
        self.dim_ring = dim_ring

        self.theta = np.linspace(0, 360, dim_ring+1)[:-1] # note that we use degree
        self.deltaTheta = self.theta[1]-self.theta[0] 
        self.choices = np.arange(dim_ring)
        self.degDiv = 360/8

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1+dim_ring+8,), dtype=np.float32)
        self.ob_dict = {'fixation': 0, 'stimulus': range(
            1, dim_ring+1), 'targetCue': range(dim_ring+1, 1+dim_ring+8)}
        self.action_space = spaces.Discrete(1+dim_ring)
        self.act_dict = {'fixation': 0, 'choice': range(1, dim_ring+1)}


    def _new_trial(self, setsize=None):
        '''
        <setsize>: int (default=None, max=8), set size (how many objects to present).
        '''
        if setsize is None:
            setsize = self.rng.choice(range(1, 8+1))

        # we do not want stimulus colors too close to each other
        rd_index = np.arange(8) * self.degDiv + \
            int(np.round(self.degDiv*0.125)) + \
            self.rng.choice(range(int(np.rint(self.degDiv * 0.75))))
        rd_index = np.floor(rd_index/2).astype('int')+1
        
        # Start with a random color wheel to sample the colors of targets
        start = np.floor(self.rng.rand()*180).astype('int')
        if start > 0 & start < 179:
            color_index = list(range(start, 180)) + list(range(start))
        elif start==0:
            color_index=list(range(180))
        elif start==179:
            color_index = [179] + list(range(179))
        color_index = np.array(color_index)
        # choose stimuli, stimulis is a list of color index between []
        stim = self.rng.choice(color_index[rd_index], setsize, replace=False)

        # choose target stimulus
        target_ind = self.rng.choice(range(setsize)) 

        # Trial info
        trial = {
            'setsize': setsize,
            'stim': stim,
            'ground_truth': stim[target_ind],
        }
        print(trial)
        ground_truth = trial['ground_truth']
        stim_theta = self.theta[ground_truth]

        # Periods
        self.add_period(['stimulus', 'delay', 'decision'])

        # Observations
        
        # add fixtion
        self.add_ob(1, period=['stimulus', 'delay'], where='fixation') # where indicate the row to add observation
        
        # generate stim input
        stimInput = np.zeros((self.dim_ring))
        for i in trial['stim']:
            stimInput += np.exp(
                2*(np.cos((self.theta - i*2)/180*np.pi)-1))


        self.add_ob(stimInput, 'stimulus', where='stimulus')
        self.add_randn(0, self.sigma, 'stimulus', where='stimulus') # add input noise
        # Todo, shall we add noise to delay period??

        # add target cue
        cueInput = np.zeros((8))
        cueInput[target_ind] = 1
        self.add_ob(cueInput, 'decision', where='targetCue')
        
        # Ground truth
        self.set_groundtruth(stim_theta, 'decision')

        return trial

    def _step(self, action):
        """
        """
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now
        # observations
        if self.in_period('stimulus') or self.in_period('delay'):
            if action != 0:  # action = 0 means fixating
                new_trial = self.abort
                reward += self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True                
                reward += self.rewards['correct']-np.abs(gt-action)/180 # smaller response error, larger rewards
                self.performance = 1
        else: # no response during decision time window
                new_trial = True
                reward += self.rewards['noresp']
            
        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}



if __name__ == '__main__':
    env = delayestimation(dt=20, timing=None)
    ngym.utils.plot_env(env, num_steps=100, def_act=1)
    # env = PerceptualDecisionMakingDelayResponse()
    # ngym.utils.plot_env(env, num_steps=100, def_act=1)
