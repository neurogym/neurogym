"""Test wrappers."""

import numpy as np
import gym
from gym import spaces
from gym.core import Wrapper

import neurogym as ngym
from neurogym.wrappers.trial_hist import TrialHistory
from neurogym.wrappers.side_bias import SideBias
from neurogym import all_tasks


def test_sidebias(env_name, verbose=False):
    env = gym.make(env_name)
    env = SideBias(env, prob=[(0, 1), (1, 0)], block_dur=10)
    env.reset()
    for stp in range(10000):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        if info['new_trial']:
            if verbose:
                print('Block', env.curr_block)
                print('Ground truth', info['gt'])
        # print(env.env.t)
        # print('')
        # print('Trial', env.unwrapped.num_tr)
        # print('Within trial time', env.unwrapped.t_ind)
        # print('Observation', obs)
        if done:
            env.reset()


def test_sidebias_all():
    """Test speed of all experiments."""
    success_count = 0
    total_count = 0
    for env_name in sorted(all_tasks.keys()):
        total_count += 1
        print('Running env: {:s} Wrapped with SideBias'.format(env_name))
        try:
            test_sidebias(env_name)
            print('Success')
            success_count += 1
        except BaseException as e:
            print('Failure at running env: {:s}'.format(env_name))
            print(e)
        print('')

    print('Success {:d}/{:d} tasks'.format(success_count, total_count))


if __name__ == '__main__':
    test_sidebias_all()