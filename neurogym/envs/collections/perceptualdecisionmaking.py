"""Specific tasks based on the same base task."""

import gymnasium as gym  # using ngym.make would lead to circular import

from neurogym import wrappers
from neurogym.utils import scheduler


def roitman02(**kwargs):
    env = gym.make("PerceptualDecisionMaking-v0", disable_env_checker=True, **kwargs)
    return wrappers.ReactionTime(env)


def ibl20(**kwargs):
    env = gym.make("PerceptualDecisionMaking-v0", disable_env_checker=True, **kwargs)
    env = wrappers.RandomGroundTruth(env)  # allow setting p of choices
    schedule = scheduler.RandomBlockSchedule(n=2, block_lens=[90, 90])
    attr_list = [{"p": (0.8, 0.2)}, {"p": (0.2, 0.8)}]  # p of each block
    return wrappers.ScheduleAttr(env, schedule, attr_list=attr_list)  # new env
