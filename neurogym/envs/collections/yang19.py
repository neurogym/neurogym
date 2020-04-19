"""An example collection of tasks."""

import gym


def yang19dm(**kwargs):
    env = gym.make('PerceptualDecisionMaking-v0', **kwargs)
    return env


def yang19ctxdm(**kwargs):
    env = gym.make('ContextDecisionMaking-v0', **kwargs)
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


if __name__ == '__main__':
    pass