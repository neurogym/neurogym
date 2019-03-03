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
    """
    Mante task
    """
    # Inputs
    inputs = tasktools.to_map('motion', 'color',
                              'm-left', 'm-right',
                              'c-left', 'c-right')
    # Actions
    actions = tasktools.to_map('fixate', 'left', 'right')

    # Trial conditions
    contexts = ['m', 'c']
    left_rights = [-1, 1]
    cohs = [5, 15, 50]

    # Input noise
    sigma = np.sqrt(2*100*0.02)

    # Rewards
    R_ABORTED = -1.
    R_CORRECT = +1.
    R_MISS = 0.
    abort = False

    # Epoch durations
    # TODO: in ms?
    fixation = 750
    stimulus = 750
    delay_min = 300
    delay_mean = 300
    delay_max = 1200
    decision = 500
    tmax = fixation + stimulus + delay_min + delay_max + decision

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, dt=50):
        # call ngm __init__ function
        super().__init__(dt=dt)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6, ),
                                            dtype=np.float32)

        self.seed()
        self.viewer = None

        

        self.trial = self._new_trial()

    # def step(rng, dt, trial, t, a):
    def step(self, action):
        # -----------------------------------------------------------------
        # Reward
        # -----------------------------------------------------------------
        trial = self.trial
        dt = self.dt
        rng = self.rng

        # epochs = trial['epochs']
        info = {'continue': True}
        reward = 0
        tr_perf = False
        if not self.in_epoch(self.t - 1, 'decision'):
            if (action != self.actions['FIXATE'] and
                    not self.in_epoch(self.t, 'fix_grace')):
                info['continue'] = not self.abort
                reward = self.R_ABORTED
        else:
            if action == self.actions['left']:
                tr_perf = True
                info['continue'] = False
                info['choice'] = 'L'
                info['t_choice'] = self.t - 1
                if trial['context'] == 'm':
                    info['correct'] = (trial['left_right_m'] < 0)
                else:
                    info['correct'] = (trial['left_right_c'] < 0)
                if info['correct']:
                    reward = self.R_CORRECT
            elif action == self.actions['right']:
                tr_perf = True
                info['continue'] = False
                info['choice'] = 'R'
                info['t_choice'] = self.t - 1
                if trial['context'] == 'm':
                    info['correct'] = (trial['left_right_m'] > 0)
                else:
                    info['correct'] = (trial['left_right_c'] > 0)
                if info['correct']:
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
                                                info['continue'],
                                                self.R_MISS, reward)

        if new_trial:
            self.t = 0
            self.num_tr += 1
            # compute perf
            self.perf, self.num_tr, self.num_tr_perf =\
                tasktools.compute_perf(self.perf, reward, self.num_tr,
                                       self.p_stp, self.num_tr_perf, tr_perf)
            self.trial = self._new_trial()
        else:
            self.t += self.dt

        self.store_data(obs, action, reward)
        done = self.num_tr > self.num_tr_exp
        return obs, reward, done, info

    def render(self, mode='human'):
        raise NotImplementedError

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
        durations = {
            'fix_grace': (0, 100),
            'fixation':  (0, self.fixation),
            'stimulus':  (self.fixation, self.fixation + self.stimulus),
            'delay':     (self.fixation + self.stimulus,
                          self.fixation + self.stimulus + delay),
            'decision':  (self.fixation + self.stimulus + delay, self.tmax),
            'tmax':      self.tmax
            }

        # -------------------------------------------------------------------------
        # Trial
        # -------------------------------------------------------------------------

        context_ = self.rng.choice(self.contexts)

        left_right_m = self.rng.choice(self.left_rights)

        left_right_c = self.rng.choice(self.left_rights)

        coh_m = self.rng.choice(self.cohs)

        coh_c = self.rng.choice(self.cohs)

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
