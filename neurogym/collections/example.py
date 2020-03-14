"""An example collection of tasks."""

import gym
from neurogym.wrappers import PassAction
from neurogym.tests.test_envs import test_run


def main():
    envs = list()
    dt = 100

    env_kwargs = {'dt': dt}
    env = gym.make('PerceptualDecisionMaking-v0', **env_kwargs)
    envs.append(env)

    env_kwargs = {'dt': dt}
    env = gym.make('ContextDecisionMaking-v0', **env_kwargs)
    envs.append(env)

    env_kwargs = {'dt': dt}
    env = gym.make('MultiSensoryIntegration-v0', **env_kwargs)
    envs.append(env)

    env_kwargs = {'dt': dt}
    env = gym.make('DelayedComparison-v0', **env_kwargs)
    envs.append(env)

    env_kwargs = {'dt': dt}
    env = gym.make('DelayedMatchSample-v0', **env_kwargs)
    envs.append(env)

    env_kwargs = {'dt': dt}
    env = gym.make('DelayedMatchCategory-v0', **env_kwargs)
    envs.append(env)

    env_kwargs = {'dt': dt}
    env = gym.make('ReachingDelayResponse-v0', **env_kwargs)
    envs.append(env)

    env_kwargs = {'dt': dt}
    env = gym.make('AntiReach-v0', **env_kwargs)
    envs.append(env)
    
    return envs


if __name__ == '__main__':
    envs = main()
    for env in envs:
        test_run(env)
