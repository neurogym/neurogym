#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tone detection auditory decision-making task."""

import numpy as np
from gym import spaces

import neurogym as ngym


class tonedetection(ngym.TrialEnv):
    '''
    By Ru-Yuan Zhang (ruyuanzhang@gmail.com)
    A subject is asked to report whether a pure tone is embeddied within a background noise

    Args:
        <dt>: delta time
        <sigma>: float, input noise level,
    '''

    metadata = {
        'paper_link': '',
        'paper_name': '',
        'tags': ['auditory', 'perceptual', 'supervised', 'decision']
    }

    def __init__(self, dt=25, sigma=0.2, timing=None):
        super().__init__(dt=dt)

        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'noresp': -0.1}  # need to change here

        self.timing = {
            'stimulus': 2000,
            'decision': 2000
            }
        if timing:
            self.timing.update(timing)
        self.toneTiming = [500, 1000, 1500] # ms, the onset times of a tone
        self.toneDur = 50 # ms, the duration of a tone

        self.abort = False

        self.sigals = np.linspace(0, 1, 5)[:-1] # signal strength
        self.conditions = [0, 1, 2, 3] # no tone, tone at position 1/2/3

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1+1,), dtype=np.float32)
        self.ob_dict = {'fixation': 0, 'stimulus': 1}
        self.action_space = spaces.Discrete(5)
        self.act_dict = {'fixation': 0, 'choice': range(1, 5+1)}

    def _new_trial(self, condition=None):
        '''
        <condition>: int (0/1/2/3), indicate no tone, tone at position 1/2/3
        '''
        if condition is None:
            condition = self.rng.choice(self.conditions)

        # Trial info
        trial = {
            'setsize': setsize,
            'stim': stim,
            'ground_truth': stim[target_ind],
        }

        ground_truth = trial['ground_truth']
        stim_theta = self.theta[ground_truth]

        # Periods
        self.add_period(['stimulus', 'decision'])

        # Observations
        
        # add fixtion
        self.add_ob(1, period=['stimulus'], where='fixation') # where indicate the row to add observation
        
        # generate stim input
        # define stimulus
        self.add_ob(stimInput, 'stimulus', where='stimulus')
        self.add_randn(0, self.sigma, 'stimulus') # add input noise
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
        if self.in_period('stimulus'):
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
    env = tonedetection(dt=20, timing=None)
    ngym.utils.plot_env(env, num_steps=100, def_act=1)
    # env = PerceptualDecisionMakingDelayResponse()
    # ngym.utils.plot_env(env, num_steps=100, def_act=1)
