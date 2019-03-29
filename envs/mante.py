"""
Context-dependent integration task, based on
  Context-dependent computation by recurrent dynamics in prefrontal cortex.
  V Mante, D Sussillo, KV Shinoy, & WT Newsome, Nature 2013.
  http://dx.doi.org/10.1038/nature12742

Code adapted from github.com/frsong/pyrl
"""
from __future__ import division

import numpy as np

import ngym
from gym import spaces

import tasktools


class Mante(ngym.ngym):
    def __init__(self, dt=100):
        # call ngm __init__ function
        super().__init__(dt=dt)

        # Inputs
        self.inputs = tasktools.to_map('motion', 'color',
                                       'm-left', 'm-right',
                                       'c-left', 'c-right')
        # Actions
        self.actions = tasktools.to_map('FIXATE', 'left', 'right')

        # trial conditions
        self.contexts = ['m', 'c']
        self.choices = [-1, 1]
        self.cohs = [5, 15, 50]

        # Input noise
        self.sigma = np.sqrt(2*100*0.02)

        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_MISS = 0.
        self.abort = False

        # Epoch durations
        self.fixation = 750
        self.stimulus = 750
        self.delay_min = 300
        self.delay_mean = 300
        self.delay_max = 1200
        self.decision = 500
        self.mean_trial_duration = self.fixation + self.stimulus +\
            self.delay_mean + self.decision
        print('mean trial duration: ' + str(self.mean_trial_duration) +
              ' (max num. steps: ' + str(self.mean_trial_duration/self.dt) +
              ')')
        # set action and observation space
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6, ),
                                            dtype=np.float32)
        # seeding
        self.seed()
        self.viewer = None

        self.trial = self._new_trial()

    def _step(self, action):
        # -----------------------------------------------------------------
        # Reward
        # -----------------------------------------------------------------
        trial = self.trial
        dt = self.dt
        rng = self.rng

        # epochs = trial['epochs']
        info = {'new_trial': False}
        reward = 0
        tr_perf = False
        if self.in_epoch(self.t, 'fixation'):
            if (action != self.actions['FIXATE']):
                info['new_trial'] = self.abort
                reward = self.R_ABORTED
        if self.in_epoch(self.t, 'decision'):
            if action == self.actions['left']:
                tr_perf = True
                info['new_trial'] = True
                if trial['context'] == 'm':
                    correct = (trial['left_right_m'] < 0)
                else:
                    correct = (trial['left_right_c'] < 0)
                if correct:
                    reward = self.R_CORRECT
            elif action == self.actions['right']:
                tr_perf = True
                info['new_trial'] = True
                if trial['context'] == 'm':
                    correct = (trial['left_right_m'] > 0)
                else:
                    correct = (trial['left_right_c'] > 0)
                if correct:
                    reward = self.R_CORRECT

        # -------------------------------------------------------------------------------------
        # Inputs
        # -------------------------------------------------------------------------------------

        if trial['context'] == 'm':
            context = self.inputs['motion']
        else:
            context = self.inputs['color']

        if trial['left_right_m'] < 0:
            high_m = self.inputs['m-left']
            low_m = self.inputs['m-right']
        else:
            high_m = self.inputs['m-right']
            low_m = self.inputs['m-left']

        if trial['left_right_c'] < 0:
            high_c = self.inputs['c-left']
            low_c = self.inputs['c-right']
        else:
            high_c = self.inputs['c-right']
            low_c = self.inputs['c-left']

        obs = np.zeros(len(self.inputs))
        if (self.in_epoch(self.t, 'fixation') or
            self.in_epoch(self.t, 'stimulus') or
                self.in_epoch(self.t, 'delay')):
            obs[context] = 1
        if self.in_epoch(self.t, 'stimulus'):
            obs[high_m] = self.scale(+trial['coh_m']) +\
                rng.normal(scale=self.sigma) / np.sqrt(dt)
            obs[low_m] = self.scale(-trial['coh_m']) +\
                rng.normal(scale=self.sigma) / np.sqrt(dt)
            obs[high_c] = self.scale(+trial['coh_c']) +\
                rng.normal(scale=self.sigma) / np.sqrt(dt)
            obs[low_c] = self.scale(-trial['coh_c']) +\
                rng.normal(scale=self.sigma) / np.sqrt(dt)
        # ---------------------------------------------------------------------
        # new trial?
        reward, new_trial = tasktools.new_trial(self.t, self.tmax, self.dt,
                                                info['new_trial'],
                                                self.R_MISS, reward)

        if new_trial:
            info['new_trial'] = True
            self.t = 0
            self.num_tr += 1
            # compute perf
            self.perf, self.num_tr_perf =\
                tasktools.compute_perf(self.perf, reward,
                                       self.num_tr_perf, tr_perf)
        else:
            self.t += self.dt

        done = self.num_tr > self.num_tr_exp
        return obs, reward, done, info, new_trial

    def step(self, action):
        obs, reward, done, info, new_trial = self._step(action)
        if new_trial:
            self.trial = self._new_trial()
        return obs, reward, done, info

    def close(self):
        if self.viewer:
            self.viewer.close()

    def _new_trial(self):
        # -----------------------------------------------------------------------
        # Epochs
        # -----------------------------------------------------------------------

        delay = self.delay_min +\
            tasktools.truncated_exponential(self.rng, self.dt, self.delay_mean,
                                            xmax=self.delay_max)
        # maximum duration of current trial
        self.tmax = self.fixation + self.stimulus + delay + self.decision
        durations = {
            'fixation':  (0, self.fixation),
            'stimulus':  (self.fixation, self.fixation + self.stimulus),
            'delay':     (self.fixation + self.stimulus,
                          self.fixation + self.stimulus + delay),
            'decision':  (self.fixation + self.stimulus + delay, self.tmax),
            }

        # -------------------------------------------------------------------------
        # Trial
        # -------------------------------------------------------------------------

        context_ = tasktools.choice(self.rng, self.contexts)

        left_right_m = tasktools.choice(self.rng, self.choices)

        left_right_c = tasktools.choice(self.rng, self.choices)

        coh_m = tasktools.choice(self.rng, self.cohs)

        coh_c = tasktools.choice(self.rng, self.cohs)

        return {
            'durations':    durations,
            'context':      context_,
            'left_right_m': left_right_m,
            'left_right_c': left_right_c,
            'coh_m':        coh_m,
            'coh_c':        coh_c
            }

    def scale(self, coh):
        """
        Input scaling
        """
        return (1 + coh/100)/2

    def terminate(perf):
        p_decision, p_correct = tasktools.correct_2AFC(perf)

        return p_decision >= 0.99 and p_correct >= 0.85
