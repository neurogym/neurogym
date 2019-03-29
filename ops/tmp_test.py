import gym
import sys
import numpy as np
import task_registrations
import matplotlib.pyplot as plt

kwargs = {'dt': 1}

env = gym.make('ReadySetGo-v0', **kwargs)

env.reset()
observations = []
for stp in range(10000):
    state, rew, done, info = env.step(0)  # env.action_space.sample())
    observations.append(state)

#    print(state)
#    print(info)
#    print(rew)
    # print(info)
obs = np.array(observations)
plt.figure()
plt.imshow(obs.T, aspect='auto')
plt.show()
