# TODO:
import gym
import sys
import numpy as np
import task_registrations
import trial_hist
import reaction_time
import compose
import manage_data as md
import matplotlib.pyplot as plt
# params = {'trial_dur': 5000, 'dt': 500,
#           'stim_ev': 0.1, 'rewards':(0.0, -.1, 1., -1.)}

# example code        task   num. steps  wrapper
# python test_new.py RDM-v0    1000    trial_hist
env = gym.make(sys.argv[1])
if sys.argv[3] == 'trial_hist':
    env = trial_hist.TrialHistory(env)
elif sys.argv[3] == 'reaction_time':
    env = reaction_time.ReactionTime(env)
elif sys.argv[3] == 'compose':
    env_extra = gym.make('GNG-v0')
    env = compose.compose(env, env_extra)
env = md.manage_data(env, plt_tr=False)
env.reset()
observations = []
rewards = []
actions = []
actions_end_of_trial = []
for stp in range(int(sys.argv[2])):
    action = env.action_space.sample()
    state, rew, done, info = env.step(action)
    if done:
        env.reset()

    observations.append(state)
    if 'new_trial' in info.keys():
        actions_end_of_trial.append(action)
    else:
        actions_end_of_trial.append(0)
    rewards.append(rew)
    actions.append(action)

obs = np.array(observations)
plt.figure()
plt.subplot(3, 1, 1)
plt.imshow(obs.T, aspect='auto')
plt.title('observations')
plt.subplot(3, 1, 2)
plt.plot(actions)
plt.plot(actions_end_of_trial, '--')
plt.title('actions')
plt.xlim([0, len(rewards)])
plt.subplot(3, 1, 3)
plt.plot(rewards, 'r')
plt.title('reward')
plt.xlim([0, len(rewards)])
plt.show()
