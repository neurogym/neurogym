"""An example collection of tasks."""

import numpy as np
import gym
from gym import spaces

import neurogym as ngym
from neurogym.wrappers.block import ScheduleEnvs
from neurogym.utils import scheduler
from neurogym.core import TrialWrapperV2


class _MultiModalityStimulus(TrialWrapperV2):
    """Move observation to specific modality."""
    def __init__(self, env, modality=0, n_modality=1):
        super().__init__(env)
        self.modality = modality
        if 'stimulus' not in self.task.ob_dict:
            raise KeyError('ob_dict does not have key stimulus')
        ind_stimulus = np.array(self.task.ob_dict['stimulus'])
        len_stimulus = len(ind_stimulus)
        ob_space = self.task.observation_space
        ob_shape = ob_space.shape[0] + (n_modality - 1) * len_stimulus
        self.observation_space = self.task.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(ob_shape,), dtype=ob_space.dtype)
        # Shift stimulus
        self.task.ob_dict['stimulus'] = ind_stimulus + len_stimulus * modality

    def new_trial(self, **kwargs):
        return self.env.new_trial(**kwargs)


class DelayComparison(ngym.PeriodEnv):
    """Delay comparison.

    Two-alternative forced choice task in which the subject
    has to compare two stimuli separated by a delay to decide
    which one has a higher frequency.
    """
    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0,
                 dim_ring=1):
        super().__init__(dt=dt)

        # trial conditions
        self.fpairs = [(18, 10), (22, 14), (26, 18), (30, 22), (34, 26)]

        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': ('uniform', (1500, 3000)),
            'f1': ('constant', 500),
            'delay': ('constant', 3000),
            'f2': ('constant', 500),
            'decision': ('constant', 100)}
        if timing:
            self.timing.update(timing)

        self.abort = False

        # Input scaling
        self.fall = np.ravel(self.fpairs)
        self.fmin = np.min(self.fall)
        self.fmax = np.max(self.fall)

        # action and observation space
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,),
                                            dtype=np.float32)
        self.ob_dict = {'fixation': 0, 'stimulus': 1}
        self.action_space = spaces.Discrete(3)
        self.act_dict = {'fixation': 0, 'choice': [1, 2]}

    def new_trial(self, **kwargs):
        self.trial = {
            'ground_truth': self.rng.choice(self.act_dict['choice']),
            'fpair': self.fpairs[self.rng.choice(len(self.fpairs))]
        }
        self.trial.update(kwargs)

        f1, f2 = self.trial['fpair']
        if self.trial['ground_truth'] == 2:
            f1, f2 = f2, f1
        self.trial['f1'] = f1
        self.trial['f2'] = f2

        # Periods
        periods = ['fixation', 'f1', 'delay', 'f2', 'decision']
        self.add_period(periods, after=0, last_period=True)

        self.add_ob(1, where='fixation')
        self.add_ob(self.scale_p(f1), 'f1', where='stimulus')
        self.add_ob(self.scale_p(f2), 'f2', where='stimulus')
        self.set_ob(0, 'decision')
        self.add_randn(0, self.sigma, ['f1', 'f2'])

        self.set_groundtruth(self.trial['ground_truth'], 'decision')

    def scale(self, f):
        return (f - self.fmin)/(self.fmax - self.fmin)

    def scale_p(self, f):
        return (1 + self.scale(f))/2

    def scale_n(self, f):
        return (1 - self.scale(f))/2

    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward and inputs
        # ---------------------------------------------------------------------
        new_trial = False
        gt = self.gt_now
        ob = self.ob_now
        # rewards
        reward = 0
        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward = self.rewards['correct']
                    self.performance = 1
                else:
                    reward = self.rewards['fail']

        return ob, reward, False, {'new_trial': new_trial, 'gt': gt}


def dm(modality=0, **kwargs):
    env = gym.make('PerceptualDecisionMaking-v0', **kwargs)
    env = _MultiModalityStimulus(env, modality=modality, n_modality=2)
    return env


def ctxdm(context=0, **kwargs):
    kwargs['context'] = context
    env = gym.make('SingleContextDecisionMaking-v0', **kwargs)
    return env


def multidm(**kwargs):
    env = gym.make('MultiSensoryIntegration-v0', **kwargs)
    return env


def dlydm(**kwargs):
    env = gym.make('DelayComparison-v0', **kwargs)
    return env


def dlymatchsample(**kwargs):
    envs = list()
    for modality in [0, 1]:
        timing = {'delay': ('choice', [100, 200, 400, 800])}
        env_kwargs = {'sigma': 0.5, 'timing': timing}
        env_kwargs.update(kwargs)
        env = gym.make('DelayMatchSample-v0', **env_kwargs)
        env = _MultiModalityStimulus(env, modality=modality, n_modality=2)
        envs.append(env)
    schedule = scheduler.RandomSchedule(len(envs))
    env = ScheduleEnvs(envs, schedule, env_input=False)
    return env


def dlymatchcategory(**kwargs):
    envs = list()
    for modality in [0, 1]:
        env = gym.make('DelayMatchCategory-v0', **kwargs)
        env = _MultiModalityStimulus(env, modality=modality, n_modality=2)
        envs.append(env)
    schedule = scheduler.RandomSchedule(len(envs))
    env = ScheduleEnvs(envs, schedule, env_input=False)
    return env


def _antigo(anti=True, **kwargs):
    envs = list()
    for modality in [0, 1]:
        env_kwargs = {'anti': anti, 'dim_ring': 2}
        env_kwargs.update(kwargs)
        env = gym.make('AntiReach-v0', **env_kwargs)
        env = _MultiModalityStimulus(env, modality=modality, n_modality=2)
        envs.append(env)
    schedule = scheduler.RandomSchedule(len(envs))
    env = ScheduleEnvs(envs, schedule, env_input=False)
    return env


def go(**kwargs):
    return _antigo(anti=False, **kwargs)


def anti(**kwargs):
    return _antigo(anti=True, **kwargs)


def dlygo(**kwargs):
    kwargs['timing'] = {'delay': ('constant', 500)}
    return _antigo(anti=False, **kwargs)


def dlyanti(**kwargs):
    kwargs['timing'] = {'delay': ('constant', 500)}
    return _antigo(anti=True, **kwargs)


def multitask(**kwargs):
    envs = [
        go(**kwargs),
        dlygo(**kwargs),
        anti(**kwargs),
        dlyanti(**kwargs),
        dm(modality=0, **kwargs),
        dm(modality=1, **kwargs),
        ctxdm(context=0, **kwargs),
        ctxdm(context=1, **kwargs),
        multidm(**kwargs),
        dlymatchcategory(**kwargs),
        dlymatchsample(**kwargs),
    ]
    schedule = scheduler.RandomSchedule(len(envs))
    env = ScheduleEnvs(envs, schedule, env_input=True)
    return env
