#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""auditory tone detection task."""

import numpy as np
from gym import spaces

import neurogym as ngym


class ToneDetection(ngym.TrialEnv):
    '''
    By Ru-Yuan Zhang (ruyuanzhang@gmail.com)
    A subject is asked to report whether a pure tone is embeddied within a background noise. If yes, should indicate the position of the tone. The tone lasts 50ms and could appear at the 500ms, 1000ms, and 1500ms. The tone is embbeded within noises.

    Note in this version we did not consider the fixation period as we mainly aim to model human data. 
    
    For an animal version of this task, please consider to include fixation and saccade cues. See 
    https://www.nature.com/articles/nn1386

    Note that the output labels is of shape (seq_len, batch_size). For a human perceptual task, you can simply run labels = labels[-1, :] get the final output.


    Args:
        <dt>: milliseconds, delta time,
        <sigma>: float, input noise level, control the task difficulty
        <timing>: stimulus timing
    '''

    metadata = {
        'paper_link': 'https://www.jneurosci.org/content/jneuro/5/12/3261.full.pdf',
        'paper_name': 'Representation of Tones in Noise in the Responses of Auditory Nerve Fibers  in Cats',
        'tags': ['auditory', 'perceptual', 'supervised', 'decision']
    }

    def __init__(self, dt=50, sigma=0.2, timing=None):
        super().__init__(dt=dt)
        ''' 
        Here the key variables are
        <self.toneDur>: ms, duration of the tone
        <self.toneTiming>: ms, onset of the tone
        '''
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'noresp': -0.1}  # need to change here

        self.timing = {
            'stimulus': 2000,
            'toneTiming': [500, 1000, 1500],
            'toneDur': 50,
            }
        if timing:
            self.timing.update(timing)
        
        self.toneTiming = self.timing['toneTiming']
        self.toneDur = self.timing['toneDur'] # ms, the duration of a tone
        
        assert dt <= self.toneDur, 'note dt must <= tone duration (default:50)'
        
        self.toneDurIdx = int(self.toneDur/dt) # how many data point it lasts

        self.toneTimingIdx = [int(i / dt) for i in self.toneTiming]
        self.stimArray = np.zeros(int(self.timing['stimulus']/dt))

        self.abort = False

        self.signals = np.linspace(0, 1, 5)[:-1] # signal strength
        self.conditions = [0, 1, 2, 3] # no tone, tone at position 1/2/3

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1,), dtype=np.float32)
        self.ob_dict = {'fixation': 0, 'stimulus': 1}
        self.action_space = spaces.Discrete(4)
        self.act_dict = {'fixation': 0, 'choice': range(1, 5+1)}

    def _new_trial(self, condition=None):
        '''
        <condition>: int (0/1/2/3), indicate no tone, tone at position 1/2/3
        '''
        if condition is None:
            condition = self.rng.choice(self.conditions)

        # Trial info
        trial = {
            'ground_truth': condition,
        }

        # generate tone stimulus
        stim = self.stimArray.copy()
        if condition != 0:
            stim[self.toneTimingIdx[condition-1]:self.toneTimingIdx[condition-1]+self.toneDurIdx] = 1

        ground_truth = trial['ground_truth']

        # Periods
        self.add_period(['stimulus'])

        # Observations
        
        # generate stim input
        # define stimulus
        stim = stim[:, np.newaxis] # stimulus must be at least two dimension with the 1st dimen as seq_len
        self.add_ob(stim, 'stimulus')
        self.add_randn(0, self.sigma, 'stimulus') # add input noise
        
        # Ground truth
        self.set_groundtruth(ground_truth)

        return trial

    def _step(self, action):
        """
        In this tone detection task, no need to define reward step function, just output the final choice.
        """
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now
        # # observations
        # if self.in_period('stimulus'): # start a new trial once step into decision stage       
        #          new_trial = True
        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}


if __name__ == '__main__':
    env = ToneDetection(dt=50, timing=None)
    ngym.utils.plot_env(env, num_steps=100, def_act=1)
    # env = PerceptualDecisionMakingDelayResponse()
    # ngym.utils.plot_env(env, num_steps=100, def_act=1)
