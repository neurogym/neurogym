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

    def __init__(self, dt=100, rewards=None, timing=None, stim_scale=1.,
                 max_num_reps=3, th_stage=0.7, keep_days=1,
                 trials_day=300, perf_len=30, stages=[0, 1, 2, 3, 4]):
        """
        Implements shaping for the delay-response task, in which agents
        have to integrate two stimuli and report which one is larger on
        average after a delay.
        stim_scale: Controls the difficulty of the experiment. (def: 1., float)
        max_num_reps: Maximum number of times that agent can go in a row
        to the same side during phase 0. (def: 3, int)
        th_stage: Performance threshold needed to proceed to the following
        phase. (def: 0.7, float)
        keep_days: Number of days that the agent will be kept in the same phase
        once arrived to the goal performacance. (def: 1, int)
        trials_day: Number of trials performed during one day. (def: 200, int)
        perf_len: Number of trials used to compute instantaneous performance.
        (def: 30, int)
        stages: Stages used to train the agent. (def: [0, 1, 2, 3, 4], list)
        """
        super().__init__(dt=dt)
        self.choices = [1, 2]
        # cohs specifies the amount of evidence
        # (which is modulated by stim_scale)
        self.cohs = np.array([0, 6.4, 12.8, 25.6, 51.2])*stim_scale
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
            'decision': ('constant', 1500)}
        if timing:
            self.timing.update(timing)

        self.stages = stages
        self.delay_durs = self.timing['delay'][1]

        self.r_fail = self.rewards['fail']
        self.action = 0
        self.abort = False
        self.firstcounts = True
        self.first_flag = False
        self.ind = 0
        self.ind_durs = 1
        if th_stage == -1:
            self.curr_ph = self.stages[4]
        else:
            self.curr_ph = self.stages[self.ind]
        # TODO: comment variables
        # TODO: even more descriptive names
        self.curr_perf = 0
        self.min_perf = 0.6  # TODO: no magic numbers
        self.delays_perf = 0.6
        self.trials_day = trials_day
        self.th_perf = [th_stage]*len(self.stages)  # TODO: simplify??
        self.day_perf = np.empty(trials_day)
        self.trials_counter = 0
        self.inst_perf = 0
        self.perf_len = perf_len
        self.mov_perf = np.zeros(perf_len)
        self.w_keep = [keep_days]*len(self.stages)  # TODO: simplify??
        self.days_keep = self.w_keep[self.ind]
        self.keep_stage = False
        self.action_counter = 0
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
        print(self.curr_ph)
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
            if np.abs(self.action_counter) >= self.max_num_reps:
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
            self.rewards['fail'] = self.r_fail
            self.firstcounts = True
        elif self.curr_ph == 3:
            self.rewards['fail'] = self.r_fail
            if self.inst_perf >= self.delays_perf and\
               self.ind_durs < len(self.delay_durs):
                self.ind_durs += 1
            dur = self.delay_durs[0:self.ind_durs]
            print(dur)
            print('----------')
            self.durs.update({'delay': np.random.choice(dur)})
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
            if np.sign(self.action_counter) == np.sign(new):
                self.action_counter += new
            else:
                self.action_counter = new

    def set_phase(self):

        self.day_perf[self.trials_counter] =\
            1*(self.rew == self.rewards['correct'])
        self.mov_perf[self.trials_counter % self.perf_len]
        self.trials_counter += 1

        # Instantaneous perfromace
        if self.trials_counter >= self.perf_len:
            self.inst_perf = np.mean(self.mov_perf)
            if self.inst_perf < self.min_perf and self.curr_ph == 2:
                self.curr_ph = 1

        if self.trials_counter >= self.trials_day:
            self.trials_counter = 0
            self.curr_perf = np.mean(self.day_perf)
            self.day_perf = np.zeros(self.trials_counter)
            if self.curr_perf >= self.th_perf[self.ind]:
                self.keep_stage = True

            else:
                self.keep_stage = False
                self.days_keep = self.w_keep[self.ind]

            if self.keep_stage:
                self.days_keep -= 1
                if self.days_keep <= 0 and\
                   self.curr_ph < self.stages[-1]:
                    self.ind += 1
                    self.curr_ph = self.stages[self.ind]
                    self.days_keep = self.w_keep[self.ind]
                    self.keep_stage = False

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
                    self.performance =\
                        self.rewards['fail'] == self.rewards['correct']

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
    env = CVLearning()
    ngym.utils.plot_env(env, num_steps_env=100,
                        obs_traces=['Fixation Cue', 'Stim1', 'Stim2'])
