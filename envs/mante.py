"""
Context-dependent integration task, based on
  Context-dependent computation by recurrent dynamics in prefrontal cortex.
  V Mante, D Sussillo, KV Shinoy, & WT Newsome, Nature 2013.
  http://dx.doi.org/10.1038/nature12742

Code adapted from github.com/frsong/pyrl
"""
from __future__ import division

import numpy as np

import gym
from gym import spaces, logger
from gym.utils import seeding

import tasktools


# Inputs
inputs = tasktools.to_map('MOTION', 'COLOR',
                          'MOTION-LEFT', 'MOTION-RIGHT',
                          'COLOR-LEFT', 'COLOR-RIGHT')

# Actions
actions = tasktools.to_map('FIXATE', 'CHOOSE-LEFT', 'CHOOSE-RIGHT')

# Trial conditions
contexts     = ['m', 'c']
left_rights  = [-1, 1]
cohs         = [5, 15, 50]
n_conditions = len(contexts) * (len(left_rights)*len(cohs))**2

# Training
n_gradient   = n_conditions
n_validation = 50*n_conditions

# Input noise
sigma = np.sqrt(2*100*0.02)

# Rewards
R_ABORTED = -1
R_CORRECT = +1

# Epoch durations
fixation = 750
stimulus = 750
delay_min = 300
delay_mean = 300
delay_max = 1200
decision = 500
tmax = fixation + stimulus + delay_min + delay_max + decision


def get_condition(rng, dt, context={}):
    # -----------------------------------------------------------------------
    # Epochs
    # -----------------------------------------------------------------------

    delay = context.get('delay')
    if delay is None:
        delay = delay_min + tasktools.truncated_exponential(rng, dt,
                                                            delay_mean,
                                                            xmax=delay_max)

    durations = {
        'fixation':  (0, fixation),
        'stimulus':  (fixation, fixation + stimulus),
        'delay':     (fixation + stimulus, fixation + stimulus + delay),
        'decision':  (fixation + stimulus + delay, tmax),
        'tmax':      tmax
        }
    time, epochs = tasktools.get_epochs_idx(dt, durations)

    #-------------------------------------------------------------------------------------
    # Trial
    #-------------------------------------------------------------------------------------

    context_ = context.get('context')
    if context_ is None:
        context_ = rng.choice(contexts)

    left_right_m = context.get('left_right_m')
    if left_right_m is None:
        left_right_m = rng.choice(left_rights)

    left_right_c = context.get('left_right_c')
    if left_right_c is None:
        left_right_c = rng.choice(left_rights)

    coh_m = context.get('coh_m')
    if coh_m is None:
        coh_m = rng.choice(cohs)

    coh_c = context.get('coh_c')
    if coh_c is None:
        coh_c = rng.choice(cohs)

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

# Input scaling
def scale(coh):
    return (1 + coh/100)/2


def terminate(perf):
    p_decision, p_correct = tasktools.correct_2AFC(perf)

    return p_decision >= 0.99 and p_correct >= 0.85


class Mante(gym.Env):
    """
    Mante task
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, dt=1):
        high = np.array([1])
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None

        self.steps_beyond_done = None

        self.dt = dt
        self.rng = np.random.RandomState(seed=0)  # TODO: revisit

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # def step(rng, dt, trial, t, a):
    def step(self, action):
        # -------------------------------------------------------------------------------------
        # Reward
        # -------------------------------------------------------------------------------------

        trial = self.trial
        dt = self.dt
        rng = self.rng

        epochs = trial['epochs']
        status = {'continue': True}
        reward = 0
        if self.t - 1 not in epochs['decision']:
            if action != actions['FIXATE']:
                status['continue'] = False
                reward = R_ABORTED
        elif self.t - 1 in epochs['decision']:
            if action == actions['CHOOSE-LEFT']:
                status['continue'] = False
                status['choice'] = 'L'
                status['t_choice'] = self.t - 1
                if trial['context'] == 'm':
                    status['correct'] = (trial['left_right_m'] < 0)
                else:
                    status['correct'] = (trial['left_right_c'] < 0)
                if status['correct']:
                    reward = R_CORRECT
            elif action == actions['CHOOSE-RIGHT']:
                status['continue'] = False
                status['choice'] = 'R'
                status['t_choice'] = self.t - 1
                if trial['context'] == 'm':
                    status['correct'] = (trial['left_right_m'] > 0)
                else:
                    status['correct'] = (trial['left_right_c'] > 0)
                if status['correct']:
                    reward = R_CORRECT

        # -------------------------------------------------------------------------------------
        # Inputs
        # -------------------------------------------------------------------------------------

        if trial['context'] == 'm':
            context = inputs['MOTION']
        else:
            context = inputs['COLOR']

        if trial['left_right_m'] < 0:
            high_m = inputs['MOTION-LEFT']
            low_m = inputs['MOTION-RIGHT']
        else:
            high_m = inputs['MOTION-RIGHT']
            low_m = inputs['MOTION-LEFT']

        if trial['left_right_c'] < 0:
            high_c = inputs['COLOR-LEFT']
            low_c = inputs['COLOR-RIGHT']
        else:
            high_c = inputs['COLOR-RIGHT']
            low_c = inputs['COLOR-LEFT']

        u = np.zeros(len(inputs))
        if self.t in epochs['fixation'] or self.t in epochs['stimulus'] or self.t in epochs['delay']:
            u[context] = 1
        if self.t in epochs['stimulus']:
            u[high_m] = scale(+trial['coh_m']) + rng.normal(scale=sigma) / np.sqrt(dt)
            u[low_m] = scale(-trial['coh_m']) + rng.normal(scale=sigma) / np.sqrt(dt)
            u[high_c] = scale(+trial['coh_c']) + rng.normal(scale=sigma) / np.sqrt(dt)
            u[low_c] = scale(-trial['coh_c']) + rng.normal(scale=sigma) / np.sqrt(dt)

        # -------------------------------------------------------------------------------------

        self.t += 1

        return u, reward, status, {}

    def step_obsolete(self, action):
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
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1

        if self.observe_state:
            obs = np.array([self.state])
        else:
            obs = np.array([self.thirst])

        return obs, reward, done, {}

    def reset(self):
        trial = get_condition(self.rng, self.dt)
        self.trial = trial
        self.t = 0

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        if self.viewer: self.viewer.close()
