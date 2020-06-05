"""An example collection of tasks."""

import numpy as np
import gym

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
        self.task.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(ob_shape,), dtype=ob_space.dtype)
        # Shift stimulus
        self.task.ob_dict['stimulus'] = ind_stimulus + len_stimulus * modality

    def new_trial(self, **kwargs):
        return self.env.new_trial(**kwargs)


def yang19dm(modality=0, **kwargs):
    env = gym.make('PerceptualDecisionMaking-v0', **kwargs)
    env = _MultiModalityStimulus(env, modality=modality, n_modality=2)
    return env


def yang19ctxdm(context=0, **kwargs):
    kwargs['context'] = context
    env = gym.make('SingleContextDecisionMaking-v0', **kwargs)
    return env


def yang19multidm(**kwargs):
    env = gym.make('MultiSensoryIntegration-v0', **kwargs)
    return env


def yang19dlydm(**kwargs):
    env = gym.make('DelayComparison-v0', **kwargs)
    return env


def yang19dlymatchsample(**kwargs):
    timing = {'delay': ('choice', [100, 200, 400, 800])}
    env_kwargs = {'sigma': 0.5, 'timing': timing}
    env_kwargs.update(kwargs)
    env = gym.make('DelayMatchSample-v0', **env_kwargs)
    return env


def yang19dlymatchcategory(**kwargs):
    env = gym.make('DelayMatchCategory-v0', **kwargs)
    return env


def yang19antigo(**kwargs):
    env = gym.make('AntiReach-v0', **kwargs)
    return env


def yang19multitask(**kwargs):
    envs = [
        yang19dm(modality=0, **kwargs),
        yang19dm(modality=1, **kwargs),
        yang19ctxdm(context=0, **kwargs),
        yang19ctxdm(context=1, **kwargs),
        yang19multidm(**kwargs)
    ]
    schedule = scheduler.RandomSchedule(len(envs))
    env = ScheduleEnvs(envs, schedule, env_input=True)
    return env


if __name__ == '__main__':
    pass