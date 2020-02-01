"""Test utilities."""

import gym
import neurogym as ngym
from neurogym.utils.data import Dataset


def _test_dataset(env):
    dataset = Dataset(env)
    for i in range(10):
        data = dataset()


def test_dataset(env, verbose=False):
    """Main function for testing if an environment is healthy."""
    if isinstance(env, str):
        print('Testing Environment:', env)
        kwargs = {'dt': 20}
        env = gym.make(env, **kwargs)
    else:
        if not isinstance(env, gym.Env):
            raise ValueError('env must be a string or a gym.Env')
    _test_dataset(env)
    if verbose:
        print(env)
    return env


def test_dataset_all(verbose_success=False):
    """Test if all environments can at least be run."""
    success_count = 0
    total_count = 0
    supervised_count = 0
    for env_name in sorted(ngym.all_envs()):
        total_count += 1

        print('Running env: {:s}'.format(env_name))
        try:
            env = test_dataset(env_name, verbose=verbose_success)
            print('Success')
            success_count += 1
            supervised_count += 'supervised' in env.metadata.get('tags', [])
        except BaseException as e:
            print('Failure at running env: {:s}'.format(env_name))
            print(e)

    print('Success {:d}/{:d} envs'.format(success_count, total_count))
    print('Expect {:d} envs to support supervised learning'.format(supervised_count))


if __name__ == '__main__':
    test_dataset_all()
