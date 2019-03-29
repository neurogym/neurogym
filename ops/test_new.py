# TODO:
import gym
import numpy as np
import trial_hist
import reaction_time
import combine
import pass_reward
import manage_data as md

import sys
import task_registrations
import matplotlib
display_mode = True
if display_mode:
    matplotlib.use('Qt5Agg')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
dt = 80
# calling from the terminal (ploting mode ON)
# example              task   num. steps                wrapper              pass_reward        plot
# python  test_new.py RDM-v0    400        trial_hist/reaction_time/combine   True/False    True/False
if len(sys.argv) > 1:
    params = {'task': sys.argv[1], 'num_steps': sys.argv[2],
              'wrapper': sys.argv[3], 'pass_reward': sys.argv[4],
              'plot': sys.argv[5]}
else:
    params = {'task': 'RDM-v0', 'num_steps': 100, 'wrapper': '',
              'pass_reward': 'True', 'plot': 'False'}

task_registrations.register_neuroTask(params['task'])
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

if params['pass_reward'] == 'True':
    env = pass_reward.PassReward(env)

# save/render data wrapper
env = md.manage_data(env, plt_tr=False)
env.seed(0)
env.action_space.seed(0)
env.reset()
observations = []
rewards = []
actions = []
actions_end_of_trial = []
for stp in range(int(params['num_steps'])):
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    if done:
        env.reset()
    print(obs)
    observations.append(obs)
    if 'new_trial' in info.keys():
        actions_end_of_trial.append(action)
    else:
        actions_end_of_trial.append(-1)
    rewards.append(rew)
    actions.append(action)

if params['plot'] == 'True':
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
