import gym
import sys
import numpy as np
import task_registrations
import matplotlib.pyplot as plt
params = {'trial_dur': 5000, 'dt': 100, 'stim_ev': 0.1, 'rewards':(0.0, -.1, 1., -1.)}
env = gym.make(sys.argv[1], **params)

env.reset()
observations = []
for stp in range(int(sys.argv[2])):
    state, rew, done, info = env.step(0)  # env.action_space.sample())
    observations.append(state)

#    print(state)
#    print(status)
#    print(rew)
    # print(info)
obs = np.array(observations)
plt.figure()
plt.imshow(obs.T, aspect='auto')
plt.show()
