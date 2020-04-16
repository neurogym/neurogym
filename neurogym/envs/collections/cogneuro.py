import gym
from neurogym.wrappers import ReactionTime


def roitman02(**kwargs):
    env = gym.make('PerceptualDecisionMaking-v0', **kwargs)
    env = ReactionTime(env)
    return env
