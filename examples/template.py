#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Example template for contributing new tasks."""

import numpy as np
from gym import spaces
import neurogym as ngym


class YourTask(ngym.PeriodEnv):  # TIP: if task has periods (alt.: ngym.TrialEnv)
    def __init__(self, dt=100, rewards=None, timing=None, sigma=1):
        super().__init__(dt=dt)
        # Possible decisions at the end of the trial
        self.choices = [1, 2]  # e.g. [left, right]
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Optional rewards dictionary
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        # Optional timing dictionary
        # if provided, self.add_period can infer timing directly
        self.timing = {
            'fixation': ('constant', 100),
            'stimulus': ('constant', 2000),
            'delay': ('constant', 0),
            'decision': ('constant', 100)}
        if timing:
            self.timing.update(timing)

        # Similar to gym envs, define observations_space and action_space
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(3,), dtype=np.float32)
        # Optional annotation of the observation space
        self.ob_dict = {'fixation': 0, 'stimulus': [1, 2]}
        self.action_space = spaces.Discrete(3)
        # Optional annotation of the action space
        self.act_dict = {'fixation': 0, 'choice': [1, 2]}

    def _new_trial(self, **kwargs):
        """
        self._new_trial() is called internally to generate a next trial.

        Typically, you need to
            set self.trial: a dictionary of trial information
            run self.add_period():
                will add time periods to the trial
                accesible through dict self.start_t and self.end_t
            run self.add_ob():
                will add observation to np array self.ob
            run self.set_groundtruth():
                will set groundtruth to np array self.gt
        """

        # Setting trial information
        self.trial = {'ground_truth': self.rng.choice(self.choices)}
        self.trial.update(kwargs)  # allows wrappers to modify the trial
        ground_truth = self.trial['ground_truth']

        # Adding periods sequentially
        self.add_period(['fixation', 'stimulus', 'delay', 'decision'])

        # Setting observations, default all 0
        # Setting fixation cue to 1 before decision period
        self.add_ob(1, where='fixation')
        self.set_ob(0, 'decision', where='fixation')
        # Set the stimulus
        stim = [0, 0, 0]
        stim[ground_truth] = 1
        self.add_ob(stim, 'stimulus')
        # adding gaussian noise to stimulus with std = self.sigma
        self.add_randn(0, self.sigma, 'stimulus', where='stimulus')

        # Setting ground-truth value for supervised learning
        self.set_groundtruth(ground_truth, 'decision')

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
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now
        # Example structure
        if not self.in_period('decision'):
            if action != 0:  # if fixation break
                reward = self.rewards['abort']
        else:
            if action != 0:
                new_trial = True
                if action == gt:  # if correct
                    reward = self.rewards['correct']
                else:  # if incorrect
                    reward = self.rewards['fail']

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}


if __name__ == '__main__':
    # Instantiate the task
    env = YourTask()
    env.new_trial()
    print('Trial observation shape', env.ob.shape)
    print('Trial action shape', env.gt.shape)
    env.reset()
    ob, reward, done, info = env.step(env.action_space.sample())
    print('Single time step observation shape', ob.shape)
