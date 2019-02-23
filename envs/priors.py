#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 08:52:21 2019

@author: molano
"""

import numpy as np
import ngym
from gym import spaces


class Priors(ngym.ngym):
    """
    two-alternative forced choice task where the probability of repeating the
    previous choice is parametrized
    """
    def __init__(self, dt=0.1, trial_dur=5, exp_dur=10**4, rep_prob=(.2, .8),
                 rewards=(0.1, -0.1, 1.0, -1.0), block_dur=200, stim_ev=0.5):
        # call ngm __init__ function
        super().__init__(dt=dt)

        # time step
        self.dt = dt
        # duration of the experiment (in num of trials)
        self.exp_dur = exp_dur
        # num actions
        self.num_actions = 3
        # num steps per trial (trial_dur input is provided in seconds)
        self.trial_dur = trial_dur / self.dt
        # rewards given for: stop fixating, keep fixating, correct, wrong
        self.rewards = rewards
        # number of trials per blocks
        self.block_dur = block_dur
        # stimulus evidence: one stimulus is always drawn from a distribution
        # N(1,1), the mean of the second stimulus is drawn from a uniform
        # distrib.=U(1-stim_ev,1). Thus the lower stim_ev is the more difficult
        # will be the task
        self.stim_ev = stim_ev
        # prob. of repeating the stimuli in the positions of previous trial
        self.rep_prob = rep_prob
        # position of the first stimulus
        self.stms_pos_new_trial = np.random.choice([0, 1])
        # keeps track of the repeating prob of the current block
        self.curr_rep_prob = np.random.choice([0, 1])
        # initialize ground truth state [stim1 mean, stim2 mean, fixation])
        # the network has to output the action corresponding to the stim1 mean
        # that will be always 1.0 (I just initialize here at 0 for convinience)
        self.int_st = np.array([0, 0, -1])
        # accumulated evidence
        self.evidence = 0
        # number of trials
        self.num_tr = 0
        # action space
        self.action_space = spaces.Discrete(3)
        # observation space
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3, 1),
                                            dtype=np.float32)

        print('--------------- Priors experiment ---------------')
        print('Duration of each trial (in steps): ' + str(self.trial_dur))
        print('Rewards: ' + str(self.rewards))
        print('Duration of each block (in trials): ' + str(self.block_dur))
        print('Repeating probabilities of each block: ' +
              str(self.rep_prob))
        print('Stim evidence: ' + str(self.stim_ev))
        print('--------------- ----------------- ---------------')

    def step(self, action):
        """""
        receives an action (fixate/right/left) and returns a new state,
        a reward, a flag variable indicating whether the experiment has ended
        and a dictionary with relevant information
        """
        new_trial = True
        done = False
        # decide which reward and state (new_trial, correct) we are in
        if self.t < self.trial_dur:
            if (self.int_st[action] != -1).all():
                reward = self.rewards[0]
            else:
                # don't abort the trial even if the network stops fixating
                reward = self.rewards[1]

            new_trial = False

        else:
            if (self.int_st[action] == 1.0).all():
                reward = self.rewards[2]
            else:
                reward = self.rewards[3]

        if new_trial:
            new_st, info = self._new_trial()
            # check if it is time to update the network
            done = self.num_tr >= self.exp_dur
        else:
            new_st = self._get_state()
            info = {}

        return new_st, reward, done, info

    def reset(self):
        state, _ = self._new_trial()
        return state

    def _get_state(self):
        """
        Outputs a new observation using stim 1 and 2 means.
        It also outputs a fixation signal that is always -1 except at the
        end of the trial that is 0
        """
        self.t += 1
        # if still in the integration period present a new observation
        if self.t < self.trial_dur:
            self.state = [np.random.normal(self.int_st[0]),
                          np.random.normal(self.int_st[1]), -1]
        else:
            self.state = [0, 0, 0]

        # update evidence
        self.evidence += self.state[0]-self.state[1]

        return np.reshape(self.state, [1, self.num_actions, 1])

    def _new_trial(self):
        """
        this function creates a new trial, deciding the amount of coherence
        (through the mean of stim 2) and the position of stim 1. Once it has
        done this it calls _get_state to get the first observation of the trial
        """
        self.num_tr += 1
        self.t = 0
        self.evidence = 0
        # this are the means of the two stimuli
        stim1 = 1.0
        stim2 = np.random.uniform(1-self.stim_ev, 1)
        assert stim2 != 1.0
        self.choices = [stim1, stim2]

        # decide the position of the stims
        # if the block is finished update the prob of repeating
        if self.num_tr % self.block_dur == 0:
            self.curr_rep_prob = int(not self.curr_rep_prob)

        # flip a coin
        repeat = np.random.uniform() < self.rep_prob[self.curr_rep_prob]
        if not repeat:
            self.stms_pos_new_trial = not(self.stms_pos_new_trial)

        aux = [self.choices[x] for x in [int(self.stms_pos_new_trial),
                                         int(not self.stms_pos_new_trial)]]

        self.int_st = np.concatenate((aux, np.array([-1])))

        # get state
        s = self._get_state()
        # store in info the correct response and stimulus evidence
        info = {'ground_truth': self.int_st, 'mean_diff': stim1-stim2}

        return s, info
