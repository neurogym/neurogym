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


class DelayMatch1DResponse(ngym.PeriodEnv):
    r"""Delay match-to-sample or category task.

    A sample stimulus is followed by a delay and test. Agents are required
    to indicate if the sample and test are in the same category.

    Args:
        matchto: str, 'sample' or 'category'
        matchgo: bool,
            if True (False), go to the last stimulus if match (non-match)
    """
    metadata = {
        'paper_link': 'https://www.nature.com/articles/nature05078',
        'paper_name': '''Experience-dependent representation
        of visual categories in parietal cortex''',
        'tags': ['perceptual', 'working memory', 'two-alternative',
                 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0,
                 dim_ring=2, matchto='sample', matchgo=True):
        super().__init__(dt=dt)
        self.matchto = matchto
        if self.matchto not in ['sample', 'category']:
            raise ValueError('Match has to be either sample or category')
        self.matchgo = matchgo
        self.choices = ['match', 'non-match']  # match, non-match

        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': ('constant', 300),
            'sample': ('constant', 500),
            'delay': ('constant', 1000),
            'test': ('constant', 500),
            'decision': ('constant', 900)}
        if timing:
            self.timing.update(timing)

        self.abort = False

        if np.mod(dim_ring, 2) != 0:
            raise ValueError('dim ring should be an even number')
        self.dim_ring = dim_ring
        self.half_ring = int(self.dim_ring/2)
        self.theta = np.linspace(0, 2 * np.pi, dim_ring + 1)[:-1]
        # Category 0 and 1
        self.i_theta0 = np.arange(int(dim_ring / 2))
        self.theta1 = self.theta[int(dim_ring / 2):]

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1 + dim_ring,), dtype=np.float32)
        self.ob_dict = {'fixation': 0, 'stimulus': range(1, dim_ring + 1)}
        self.action_space = spaces.Discrete(1+dim_ring)
        self.act_dict = {'fixation': 0, 'choice': range(1, dim_ring+1)}

    def new_trial(self, **kwargs):
        # Trial info
        self.trial = {
            'ground_truth': self.rng.choice(self.choices),
        }
        self.trial.update(**kwargs)

        ground_truth = self.trial['ground_truth']
        i_sample_theta = self.rng.choice(self.dim_ring)
        if self.matchto == 'category':
            sample_category = (i_sample_theta > self.half_ring) * 1
            if ground_truth == 'match':
                test_category = sample_category
            else:
                test_category = 1 - sample_category
            i_test_theta = self.rng.choice(self.half_ring)
            i_test_theta += test_category * self.half_ring
        else:  # match to sample
            if ground_truth == 'match':
                i_test_theta = i_sample_theta
            else:
                # non-match is 180 degree apart
                i_test_theta = np.mod(
                    i_sample_theta + self.half_ring, self.dim_ring)

        self.trial['sample_theta'] = sample_theta = self.theta[i_sample_theta]
        self.trial['test_theta'] = test_theta = self.theta[i_test_theta]

        stim_sample = np.cos(self.theta - sample_theta) * 0.5 + 0.5
        stim_test = np.cos(self.theta - test_theta) * 0.5 + 0.5

        # Periods
        self.add_period(['fixation', 'sample', 'delay', 'test', 'decision'],
                        after=0, last_period=True)

        self.add_ob(1, where='fixation')
        self.set_ob(0, 'decision', where='fixation')
        self.add_ob(stim_sample, 'sample', where='stimulus')
        self.add_ob(stim_test, 'test', where='stimulus')
        self.add_randn(0, self.sigma, ['sample', 'test'], where='stimulus')

        if ((ground_truth == 'match' and self.matchgo) or
                (ground_truth == 'non-match' and not self.matchgo)):
            self.set_groundtruth(self.act_dict['choice'][i_test_theta], 'decision')

    def _step(self, action, **kwargs):
        new_trial = False

        obs = self.ob_now
        gt = self.gt_now

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

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt}


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


def _dlymatch(matchto, matchgo, **kwargs):
    envs = list()
    for modality in [0, 1]:
        env_kwargs = {'matchto': matchto, 'matchgo': matchgo}
        env_kwargs.update(kwargs)
        env = DelayMatch1DResponse(**env_kwargs)
        env = _MultiModalityStimulus(env, modality=modality, n_modality=2)
        envs.append(env)
    schedule = scheduler.RandomSchedule(len(envs))
    env = ScheduleEnvs(envs, schedule, env_input=False)
    return env


def dms(**kwargs):
    return _dlymatch(matchto='sample', matchgo=True, **kwargs)


def dnms(**kwargs):
    return _dlymatch(matchto='sample', matchgo=False, **kwargs)


def dmc(**kwargs):
    return _dlymatch(matchto='category', matchgo=True, **kwargs)


def dnmc(**kwargs):
    return _dlymatch(matchto='category', matchgo=False, **kwargs)


def _antigo(anti=True, **kwargs):
    envs = list()
    for modality in [0, 1]:
        env_kwargs = {'anti': anti}
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
    kwargs['dim_ring'] = 16
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
        dms(**kwargs),
        dnms(**kwargs),
        dmc(**kwargs),
        dnmc(**kwargs),
    ]
    schedule = scheduler.RandomSchedule(len(envs))
    env = ScheduleEnvs(envs, schedule, env_input=True)
    return env
