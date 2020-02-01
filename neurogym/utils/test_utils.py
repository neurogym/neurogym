"""Test utilities."""

import gym
import neurogym as ngym
from neurogym.utils.data import Dataset


def test_dataset(env):
    """Main function for testing if an environment is healthy."""
    print('Testing Environment:', env)
    kwargs = {'dt': 20}
    dataset = Dataset(env, env_kwargs=kwargs, batch_size=16, seq_len=300,
                      cache_len=1e4)
    for i in range(10):
        inputs, target = dataset()
        assert inputs.shape[0] == target.shape[0]


def test_dataset_all():
    """Test if all environments can at least be run."""
    success_count = 0
    total_count = 0
    supervised_count = len(ngym.all_envs(tag='supervised'))
    for env_name in sorted(ngym.all_envs()):
        total_count += 1

        print('Running env: {:s}'.format(env_name))
        try:
            test_dataset(env_name)
            print('Success')
            success_count += 1
        except BaseException as e:
            print('Failure at running env: {:s}'.format(env_name))
            print(e)

    print('Success {:d}/{:d} envs'.format(success_count, total_count))
    print('Expect {:d} envs to support supervised learning'.format(supervised_count))


if __name__ == '__main__':
    test_dataset_all()
