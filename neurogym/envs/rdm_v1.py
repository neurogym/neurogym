#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Random dot motion task.

TODO: Add paper
"""

import numpy as np
from gym import spaces

import neurogym as ngym
from neurogym.ops import tasktools


def get_default_timing():
    return {'fixation': ('constant', (500,)),
            'stimulus': ('truncated_exponential', [330, 80, 1500]),
            'decision': ('constant', (500,))}


class RDM(ngym.EpochEnv):
    def __init__(self, dt=100, timing=None, stimEv=1., **kwargs):
        super().__init__(dt=dt)
        self.choices = [1, 2]
        # cohs specifies the amount of evidence (which is modulated by stimEv)
        self.cohs = np.array([0, 6.4, 12.8, 25.6, 51.2]) * stimEv
        # Input noise
        self.sigma = np.sqrt(2 * 100 * 0.01)
        self.sigma_dt = self.sigma / np.sqrt(self.dt)

        default_timing = get_default_timing()
        if timing is not None:
            default_timing.update(timing)
        self.set_epochtiming(default_timing)

        # self.mean_trial_duration = self.fixation + self.stimulus_mean + \
        #                            self.decision
        # # TODO: How to make this easier?
        # self.max_trial_duration = self.fixation + self.stimulus_max + \
        #                           self.decision
        # self.max_steps = int(self.max_trial_duration / dt)

        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_FAIL = 0.
        self.R_MISS = 0.
        self.abort = False
        # action and observation spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)

    def __str__(self):
        string = ''
        if (self.fixation == 0 or self.decision == 0 or
                self.stimulus_mean == 0):
            string += 'XXXXXXXXXXXXXXXXXXXXXX\n'
            string += 'the duration of all periods must be larger than 0\n'
            string += 'XXXXXXXXXXXXXXXXXXXXXX\n'
        string += 'XXXXXXXXXXXXXXXXXXXXXX\n'
        string += 'Random Dots Motion Task\n'
        string += 'Mean Fixation: ' + str(self.fixation) + '\n'
        string += 'Min Stimulus Duration: ' + str(self.stimulus_min) + '\n'
        string += 'Mean Stimulus Duration: ' + str(self.stimulus_mean) + '\n'
        string += 'Max Stimulus Duration: ' + str(self.stimulus_max) + '\n'
        string += 'Decision: ' + str(self.decision) + '\n'
        string += '(time step: ' + str(self.dt) + '\n'
        string += 'XXXXXXXXXXXXXXXXXXXXXX\n'
        return string

    def _new_trial(self, **kwargs):
        """
        new_trial() is called when a trial ends to generate the next trial.
        The following variables are created:
            durations, which stores the duration of the different periods (in
            the case of rdm: fixation, stimulus and decision periods)
            ground truth: correct response for the trial
            coh: stimulus coherence (evidence) for the trial
            obs: observation
        """
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        ground_truth = self.rng.choice(self.choices)
        coh = self.rng.choice(self.cohs)
        self.ground_truth = ground_truth
        self.coh = coh

        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        self.add_epoch('fixation', after=0)
        self.add_epoch('stimulus', after='fixation')
        self.add_epoch('decision', after='stimulus', last_epoch=True)

        self.set_ob('fixation', [1, 0, 0])
        if ground_truth == 1:
            self.set_ob('stimulus', [1,  (1 + coh / 100) / 2, (1 - coh / 100) / 2])
        else:
            self.set_ob('stimulus', [1,  (1 - coh / 100) / 2, (1 + coh / 100) / 2])

        self.obs[self.stimulus_ind0:self.stimulus_ind1] += np.random.randn(
            *self.obs[self.stimulus_ind0:self.stimulus_ind1].shape) * self.sigma_dt

        self.set_groundtruth('decision', ground_truth)

    def _step(self, action):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """
        # ---------------------------------------------------------------------
        # Reward and observations
        # ---------------------------------------------------------------------
        new_trial = False
        # rewards
        reward = 0
        obs = self.obs[self.t_ind]
        gt = self.gt[self.t_ind]
        # observations
        if self.in_epoch('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.R_ABORTED
        elif self.in_epoch('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward = self.R_CORRECT
                else:
                    reward = self.R_FAIL

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt}



if __name__ == '__main__':
    env = RDM()
    tasktools.plot_struct(env, num_steps_env=50000)
