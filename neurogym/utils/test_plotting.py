import numpy as np
import matplotlib.pyplot as plt

import gym


def test_plot(env_name, num_steps=500, kwargs={'dt': 100}):
    env = gym.make(env_name, **kwargs)

    env.reset()
    observations = []
    for stp in range(num_steps):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        observations.append(obs)

        # print(state)
        # print(info)
        # print(rew)
        # print(info)
    observations = np.array(observations)
    plt.figure()
    plt.imshow(observations.T, aspect='auto')
    plt.show()
