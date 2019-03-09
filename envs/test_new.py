# TODO:
import gym
import sys
import numpy as np
import task_registrations
import trial_hist
import reaction_time
import combine
import manage_data as md
import matplotlib.pyplot as plt
dt = 80
# calling from the terminal
# example            task   num. steps  wrapper
# python test_new.py RDM-v0    1000    trial_hist
if len(sys.argv) > 0:
    params = {'task': sys.argv[1], 'num_steps': sys.argv[2],
              'wrapper': sys.argv[3], 'plot': len(sys.argv) > 4}
else:
    params = {'task': 'GNG-v0', 'num_steps': 1000, 'wrapper': '', 'plot': True}

# task
env = gym.make(params['task'], **{'dt': dt})
# wrappers
if params['wrapper'] == 'trial_hist':
    env = trial_hist.TrialHistory(env)
elif params['wrapper'] == 'reaction_time':
    env = reaction_time.ReactionTime(env)
elif params['wrapper'] == 'combine':
    env_extra = gym.make('GNG-v0', **{'dt': dt})
    # delay is in ms
    env = combine.combine(dt=dt, env1=env, env2=env_extra, delay=200)

# save/render data wrapper
env = md.manage_data(env, plt_tr=False)
env.reset()
observations = []
rewards = []
actions = []
actions_end_of_trial = []
for stp in range(int(params['num_steps'])):
    action = env.action_space.sample()
    state, rew, done, info = env.step(action)
    if done:
        env.reset()

    observations.append(state)
    if 'new_trial' in info.keys():
        actions_end_of_trial.append(action)
    else:
        actions_end_of_trial.append(-1)
    rewards.append(rew)
    actions.append(action)

if params['plot']:
    obs = np.array(observations)
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.imshow(obs.T, aspect='auto')
    plt.title('observations')
    plt.subplot(3, 1, 2)
    plt.plot(actions, marker='+')
    plt.plot(actions_end_of_trial, '--')
    plt.title('actions')
    plt.xlim([0, len(rewards)])
    plt.subplot(3, 1, 3)
    plt.plot(rewards, 'r')
    plt.title('reward')
    plt.xlim([0, len(rewards)])
    plt.show()
