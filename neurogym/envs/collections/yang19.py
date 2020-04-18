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
    env = gym.make('DelayedComparison-v0', **kwargs)
    return env


def yang19dlymatchsample(**kwargs):
    env = gym.make('DelayedMatchSample-v0', **kwargs)
    return env


def yang19dlymatchcategory(**kwargs):
    env = gym.make('DelayedMatchCategory-v0', **kwargs)
    return env


def yang19antigo(**kwargs):
    env = gym.make('AntiReach-v0', **kwargs)
    return env


if __name__ == '__main__':
    pass