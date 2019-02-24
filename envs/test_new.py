import gym
import sys
import task_registrations

env = gym.make(sys.argv[1])

env.reset()
observations = []
for stp in range(int(sys.argv[2])):
    state, rew, done, info = env.step(0)  # env.action_space.sample())
    observations.append(state)

#    print(state)
#    print(status)
#    print(rew)
    # print(info)
