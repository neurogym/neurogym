"""Test wrappers."""

import numpy as np
import gym
from gym import spaces
from gym.core import Wrapper

import neurogym as ngym
from neurogym.wrappers.trial_hist import TrialHistory
from neurogym.wrappers.side_bias import SideBias


def test_sidebias():
    env = gym.make('RDM-v0')
    env = SideBias(env, prob=[(0, 1), (1, 0)], block_dur=10)
    env.reset()
    for stp in range(2000):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        if info['new_trial']:
            pass
            # print('Block', env.curr_block)
            # print('Ground truth', info['gt'])
        # print(env.env.t)
        # print('')
        # print('Trial', env.unwrapped.num_tr)
        # print('Within trial time', env.unwrapped.t_ind)
        # print('Observation', obs)
        if done:
            env.reset()


if __name__ == '__main__':
    test_sidebias()