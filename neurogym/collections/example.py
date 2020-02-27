"""An example collection of tasks."""

import gym
from neurogym.wrappers import PassAction
from neurogym.tests.test_envs import test_run

env_name = 'PerceptualDecisionMaking-v0'
env_kwargs = {'dt': 20, 'stim_scale': 0.5}
env = gym.make(env_name, **env_kwargs)
env = PassAction(env)

test_run(env)

