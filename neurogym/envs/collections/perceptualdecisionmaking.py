"""Specific tasks based on the same base task."""

import gym
import neurogym.wrappers as wrappers


def roitman02(**kwargs):
    env = gym.make('PerceptualDecisionMaking-v0', **kwargs)
    env = wrappers.ReactionTime(env)
    return env
