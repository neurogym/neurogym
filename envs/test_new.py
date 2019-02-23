import gym
from gym.envs.registration import register

register(
    id='Mante-v0',
    entry_point='mante:Mante',
    max_episode_steps=100000,
    reward_threshold=90.0,
)

env = gym.make('Mante-v0')

env.reset()
for stp in range(10000):
    state, rew, status, info = env.step(0)  # env.action_space.sample())

#    print(state)
#    print(status)
#    print(rew)
#    print(info)
