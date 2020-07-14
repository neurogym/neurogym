"""Test speed of various code."""

import time

import gym
import neurogym as ngym


def test_speed(env, n_steps=100000, warmup_steps=10000):
    """Test speed of an environment."""
    if isinstance(env, str):
        kwargs = {'dt': 20}
        env = gym.make(env, **kwargs)

    env.reset()
    for stp in range(warmup_steps):
        action = env.action_space.sample()
        state, rew, done, info = env.step(action)  # env.action_space.sample())
        if done:
            env.reset()

    total_time = 0
    env.reset()
    for stp in range(n_steps):
        action = env.action_space.sample()
        start_time = time.time()
        state, rew, done, info = env.step(action)  # env.action_space.sample())
        total_time += time.time() - start_time
        if done:
            env.reset()

    print('Time/step {:0.3f}us [with stepping]'.format(total_time/n_steps*1e6))
    return env


def test_speed_with_new_trial(env):
    """Test speed of an environment."""
    n_trials = 1000
    warmup_trials = 100
    kwargs = {'dt': 20}

    if isinstance(env, str):
        env = gym.make(env, **kwargs)

    env.reset()
    for stp in range(warmup_trials):
        env.new_trial()

    n_steps = 0
    start_time = time.time()
    env.reset()
    for stp in range(n_trials):
        env.new_trial()
        n_steps += env.ob.shape[0]
    total_time = time.time() - start_time

    print('Time/step {:0.3f}us [with new trial]'.format(total_time/n_steps*1e6))
    return env


def test_speed_all():
    """Test speed of all experiments."""
    for env_name in sorted(ngym.all_envs()):
        print('Running env: {:s}'.format(env_name))
        try:
            test_speed(env_name)
            print('Success')
        except BaseException as e:
            print('Failure at running env: {:s}'.format(env_name))
            print(e)


def test_speed_dataset(env):
    batch_size = 16
    seq_len = 100
    kwargs = {}
    dataset = ngym.Dataset(
        env, env_kwargs=kwargs, batch_size=batch_size, seq_len=seq_len)
    n_batch = 100
    start_time = time.time()
    for batch in range(n_batch):
        _, _ = dataset()
    total_time = time.time() - start_time
    time_per_batch = total_time / n_batch
    time_per_step = total_time / n_batch / batch_size / seq_len
    print('Time/batch {:0.3f}us [with dataset]'.format(time_per_batch * 1e6))
    print('Time/step {:0.3f}us [with dataset]'.format(time_per_step * 1e6))


def test_speed_dataset_all():
    """Test dataset speed of all experiments."""
    for env_name in sorted(ngym.all_envs()):
        print('Running env: {:s}'.format(env_name))
        try:
            test_speed_dataset(env_name)
            print('Success')
        except BaseException as e:
            print('Failure at running env: {:s}'.format(env_name))
            print(e)


if __name__ == '__main__':
    pass