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
from gym import spaces, logger

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
    R_ABORTED = -1
    R_CORRECT = +1

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

        self.steps_beyond_done = None

        self.trial = self._new_trial(self.rng, self.dt)

    # def step(rng, dt, trial, t, a):
    def step(self, action):
        # -----------------------------------------------------------------
        # Reward
        # -----------------------------------------------------------------
        trial = self.trial
        dt = self.dt
        rng = self.rng

        epochs = trial['epochs']
        status = {'continue': True}
        reward = 0
        if self.t - 1 not in epochs['decision']:
            if action != self.actions['fixate']:
                status['continue'] = False  # TODO: abort when no fixating?
                reward = self.R_ABORTED
        elif self.t - 1 in epochs['decision']:
            if action == self.actions['left']:
                status['continue'] = False
                status['choice'] = 'L'
                status['t_choice'] = self.t - 1
                if trial['context'] == 'm':
                    status['correct'] = (trial['left_right_m'] < 0)
                else:
                    status['correct'] = (trial['left_right_c'] < 0)
                if status['correct']:
                    reward = self.R_CORRECT
            elif action == self.actions['right']:
                status['continue'] = False
                status['choice'] = 'R'
                status['t_choice'] = self.t - 1
                if trial['context'] == 'm':
                    status['correct'] = (trial['left_right_m'] > 0)
                else:
                    status['correct'] = (trial['left_right_c'] > 0)
                if status['correct']:
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
        if self.t in epochs['fixation'] or self.t in epochs['stimulus'] or\
           self.t in epochs['delay']:
            obs[context] = 1
        if self.t in epochs['stimulus']:
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
        done, self.t, self.perf = tasktools.new_trial(self.t, self.tmax,
                                                      self.dt,
                                                      status['continue'],
                                                      self.R_ABORTED,
                                                      self.num_tr % self.p_stp,
                                                      self.perf,
                                                      reward)
        return obs, reward, done, status

    def step_obsolete(self, action):  # TODO: what is this function?
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action, type(action))
        state = self.state

        if action == 0:
            reward = self.effort
            # poke
            if state <= 0:
                # reward and reset state
                reward += self.reward_seq_complete * self.thirst
                state = self.n_press
                self.thirst_state = -8
        elif action == 1:
            # press
            reward = self.effort
            state -= 1
            state = max(0, state)
        elif action == 2:
            # rest
            reward = 0.0
        else:
            raise ValueError

        self.thirst_state += self.np_random.rand() * 0.4 + 0.8
        self.thirst = self._get_thirst(self.thirst_state)
        self.state = state
        done = False

        if not done:
            pass
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this " +
                            " environment has already returned done = True." +
                            "  You should always call 'reset()' once you " +
                            " receive 'done = True' -- any further steps are" +
                            "  undefined behavior.")
            self.steps_beyond_done += 1

        if self.observe_state:
            obs = np.array([self.state])
        else:
            obs = np.array([self.thirst])

        return obs, reward, done, {}

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        if self.viewer:
            self.viewer.close()

    def _new_trial(self, rng, dt, context={}):
        # -----------------------------------------------------------------------
        # Epochs
        # -----------------------------------------------------------------------

        delay = context.get('delay')
        if delay is None:
            delay = self.delay_min +\
                tasktools.truncated_exponential(rng, dt, self.delay_mean,
                                                xmax=self.delay_max)
        durations = {
            'fixation':  (0, self.fixation),
            'stimulus':  (self.fixation, self.fixation + self.stimulus),
            'delay':     (self.fixation + self.stimulus,
                          self.fixation + self.stimulus + delay),
            'decision':  (self.fixation + self.stimulus + delay, self.tmax),
            'tmax':      self.tmax
            }
        time, epochs = tasktools.get_epochs_idx(dt, durations)

        # -------------------------------------------------------------------------
        # Trial
        # -------------------------------------------------------------------------

        context_ = context.get('context')
        if context_ is None:
            context_ = rng.choice(self.contexts)

        left_right_m = context.get('left_right_m')
        if left_right_m is None:
            left_right_m = rng.choice(self.left_rights)

        left_right_c = context.get('left_right_c')
        if left_right_c is None:
            left_right_c = rng.choice(self.left_rights)

        coh_m = context.get('coh_m')
        if coh_m is None:
            coh_m = rng.choice(self.cohs)

        coh_c = context.get('coh_c')
        if coh_c is None:
            coh_c = rng.choice(self.cohs)

        return {
            'durations':    durations,
            'time':         time,
            'epochs':       epochs,
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
