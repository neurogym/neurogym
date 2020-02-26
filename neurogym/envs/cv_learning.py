#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:45:33 2019

@author: molano
"""


import numpy as np
from gym import spaces
import neurogym as ngym


class CVLearning(ngym.PeriodEnv):
    metadata = {
        'description': 'Implements shaping for the delay-response task,' +
        ' in which agents have to integrate two stimuli and report' +
        ' which one is larger on average after a delay.',
        'paper_link': 'https://www.nature.com/articles/s41586-019-0919-7',
        'paper_name': 'Discrete attractor dynamics underlies persistent' +
        ' activity in the frontal cortex',
        'tags': ['perceptual', 'delayed response', 'two-alternative',
                 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None, stimEv=1.,
                 perf_w=100, max_num_reps=3, init_ph=0, th=0.8):
        """
        Implements shaping for the delay-response task, in which agents
        have to integrate two stimuli and report which one is larger on
        average after a delay.
        dt: Timestep duration. (def: 100 (ms), int)
        rewards: dictionary of rewards
        timing: Description and duration of periods forming a trial.
        stimEv: Controls the difficulty of the experiment. (def: 1., float)
        perf_w: Window used to compute the mean reward. (def: 1000, int)
        max_num_reps: Maximum number of times that agent can go in a row
        to the same side during phase 0. (def: 3, int)
        init_ph: Phase initializing the task. (def: 0, int)
        th: Performance threshold needed to proceed to the following phase.
        (def: 0.8, float)
        """
        super().__init__(dt=dt)
        self.choices = [1, 2]
        # cohs specifies the amount of evidence (which is modulated by stimEv)
        self.cohs = np.array([0, 6.4, 12.8, 25.6, 51.2])*stimEv
        # Input noise
        sigma = np.sqrt(2*100*0.01)
        self.sigma_dt = sigma / np.sqrt(self.dt)

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': -1.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': ('constant', 200),
            'stimulus': ('constant', 1150),
            'delay': ('choice', [300, 500, 700, 900, 1200, 2000, 3200, 4000]),
            # 'go_cue': ('constant', 100), # TODO: Not implemented
            'decision': ('constant', 1500)}
        if timing:
            self.timing.update(timing)

        self.action = 0
        self.abort = False
        self.firstcounts = True
        self.first_flag = False
        self.curr_ph = init_ph
        self.curr_perf = 0
        self.perf_window = perf_w
        self.goal_perf = [th]*4
        self.mov_window = []
        self.counter = 0
        self.max_num_reps = max_num_reps
        self.rew = 0
        # action and observation spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)

    def new_trial(self, **kwargs):
        """
        new_trial() is called when a trial ends to generate the next trial.
        The following variables are created:
            durations: Stores the duration of the different periods.
            ground truth: Correct response for the trial.
            coh: Stimulus coherence (evidence) for the trial.
            obs: Observation.
        """
        self.set_phase()
        if self.curr_ph == 0:
            # control that agent does not repeat side more than 3 times
            self.count(self.action)

        self.trial = {
            'ground_truth': self.rng.choice(self.choices),
            'coh': self.rng.choice(self.cohs),
            'sigma_dt': self.sigma_dt,
        }

        # init durations with None
        self.durs = {key: None for key in self.timing}

        self.first_choice_rew = None
        if self.curr_ph == 0:
            # no stim, reward is in both left and right
            # agent cannot go N times in a row to the same side
            if np.abs(self.counter) >= self.max_num_reps:
                ground_truth = 1 if self.action == 2 else 2
                self.trial.update({'ground_truth': ground_truth})
                self.rewards['fail'] = 0
            else:
                self.rewards['fail'] = self.rewards['correct']
            self.durs.update({'stimulus': (0),
                             'delay': (0)})
            self.trial.update({'sigma_dt': 0})

        elif self.curr_ph == 1:
            # stim introduced with no ambiguity
            # wrong answer is not penalized
            # agent can keep exploring until finding the right answer
            self.durs.update({'delay': (0)})
            self.trial.update({'coh': 100})
            self.trial.update({'sigma_dt': 0})
            self.rewards['fail'] = 0
            self.firstcounts = False
        elif self.curr_ph == 2:
            # first answer counts
            # wrong answer is penalized
            self.durs.update({'delay': (0)})
            self.trial.update({'coh': 100})
            self.trial.update({'sigma_dt': 0})
            self.rewards['fail'] = -1
            self.firstcounts = True
        elif self.curr_ph == 3:
            # delay component is introduced
            self.trial.update({'coh': 100})
            self.trial.update({'sigma_dt': 0})
        # phase 4: ambiguity component is introduced

        self.first_flag = False

        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------

        self.trial.update(kwargs)

        # ---------------------------------------------------------------------
        # Periods
        # ---------------------------------------------------------------------
        self.add_period('fixation', after=0)
        self.add_period('stimulus', duration=self.durs['stimulus'],
                        after='fixation')
        self.add_period('delay', duration=self.durs['delay'],
                        after='stimulus')
        self.add_period('decision', after='delay', last_period=True)

        # define observations
        self.set_ob([1, 0, 0], 'fixation')
        stim = self.view_ob('stimulus')
        stim[:, 0] = 1
        stim[:, 1:] = (1 - self.trial['coh']/100)/2
        stim[:, self.trial['ground_truth']] = (1 + self.trial['coh']/100)/2
        stim[:, 1:] +=\
            self.rng.randn(stim.shape[0], 2) * self.trial['sigma_dt']

        self.set_ob([1, 0, 0], 'delay')

        self.set_groundtruth(self.trial['ground_truth'], 'decision')

    def count(self, action):
        '''
        check the last three answers during stage 0 so the network has to
        alternate between left and right
        '''
        if action != 0:
            new = action - 2/action
            if np.sign(self.counter) == np.sign(new):
                self.counter += new
            else:
                self.counter = new

    def set_phase(self):
        if self.curr_ph < 4:
            if len(self.mov_window) >= self.perf_window:
                self.mov_window.append(1*(self.rew == self.rewards['correct']))
                self.mov_window.pop(0)  # remove first value
                self.curr_perf = np.sum(self.mov_window)/self.perf_window
                if self.curr_perf >= self.goal_perf[self.curr_ph]:
                    self.curr_ph += 1
                    self.mov_window = []
            else:
                self.mov_window.append(1*(self.rew == self.rewards['correct']))

    def _step(self, action):
        # obs, reward, done, info = self.env._step(action)
        # ---------------------------------------------------------------------

        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now
        first_choice = False
        if self.in_period('fixation') or self.in_period('delay'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']
        elif self.in_period('decision'):
            if action == gt:
                reward = self.rewards['correct']
                new_trial = True
                if not self.first_flag:
                    first_choice = True
                    self.first_flag = True
                    self.performance = 1
            elif action == 3 - gt:  # 3-action is the other act
                reward = self.rewards['fail']
                new_trial = self.firstcounts
                if not self.first_flag:
                    first_choice = True
                    self.first_flag = True
                    self.performance = self.rewards['fail'] == self.rewards['correct']

        # check if first choice (phase 1)
        if not self.firstcounts and first_choice:
            self.first_choice_rew = reward
        # set reward for all phases
        self.rew = self.first_choice_rew or reward

        if new_trial and self.curr_ph == 0:
            self.action = action
        info = {'new_trial': new_trial, 'gt': gt, 'num_tr': self.num_tr,
                'curr_ph': self.curr_ph, 'first_rew': self.rew}
        return self.obs_now, reward, False, info


if __name__ == '__main__':
    env = CVLearning(init_ph=0)
    ngym.utils.plot_env(env, num_steps_env=100,
                        obs_traces=['Fixation Cue', 'Stim1', 'Stim2'])
