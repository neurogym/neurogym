"""Plotting functions."""

import numpy as np
import matplotlib.pyplot as plt

import gym


def plot_env(env, num_steps_env=200, num_steps_plt=None,
             def_act=None, model=None, name=None, legend=True):
    if num_steps_plt is None:
        num_steps_plt = num_steps_env
    if isinstance(env, str):
        env = gym.make(env)
    if name is None:
        name = type(env).__name__
    observations = []
    obs_cum = []
    state_mat = []
    rewards = []
    actions = []
    actions_end_of_trial = []
    gt = []
    perf = []
    obs = env.reset()
    obs_cum_temp = obs
    for stp in range(int(num_steps_env)):
        if model is not None:
            action, _states = model.predict(obs)
            if isinstance(action, float) or isinstance(action, int):
                action = [action]
            state_mat.append(_states)
        elif def_act is not None:
            action = def_act
        else:
            action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        obs_cum_temp += obs
        obs_cum.append(obs_cum_temp.copy())
        if isinstance(info, list):
            info = info[0]
            obs_aux = obs[0]
            rew = rew[0]
            done = done[0]
            action = action[0]
        else:
            obs_aux = obs

        if done:
            env.reset()
        observations.append(obs_aux)
        if info['new_trial']:
            actions_end_of_trial.append(action)
            perf.append(rew)
            obs_cum_temp = np.zeros_like(obs_cum_temp)
        else:
            actions_end_of_trial.append(-1)
        rewards.append(rew)
        actions.append(action)
        gt.append(info['gt'])
    if model is not None:
        states = np.array(state_mat)
        states = states[:, 0, :]
    else:
        states = None
    obs_cum = np.array(obs_cum)
    obs = np.array(observations)
    fig_(obs, actions, gt, rewards, num_steps_plt, perf, legend=legend,
         states=states, name=name)
    data = {'obs': obs, 'obs_cum': obs_cum, 'rewards': rewards,
            'actions': actions, 'perf': perf,
            'actions_end_of_trial': actions_end_of_trial, 'gt': gt,
            'states': states}
    return data


def fig_(obs, actions, gt, rewards, num_steps_plt, perf, legend=True,
         obs_cum=None, states=None, name='', folder=''):
    if states is not None:
        rows = 4
    else:
        rows = 3
    gt_colors = 'rgkmc'
    f = plt.figure(figsize=(8, 8))
    # obs
    plt.subplot(rows, 1, 1)
    plt.imshow(obs[:num_steps_plt, :].T, aspect='auto')
    plt.title('observations ' + name + ' env')
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    # actions
    plt.subplot(rows, 1, 2)
    plt.plot(np.arange(num_steps_plt) + 0.,
             actions[:num_steps_plt], marker='+', label='actions')
    gt = np.array(gt)
    for ind_gt in range(gt.shape[1]):
        plt.plot(np.arange(num_steps_plt) + 0.,
                 gt[:num_steps_plt, ind_gt], gt_colors[ind_gt],
                 label='ground truth '+str(ind_gt))
    plt.ylabel('actions')
    if legend:
        plt.legend()
    plt.xlim([-0.5, num_steps_plt-0.5])
    ax = plt.gca()
    ax.set_xticks([])
    # rewards
    plt.subplot(rows, 1, 3)
    plt.plot(np.arange(num_steps_plt) + 0.,
             rewards[:num_steps_plt], 'r')
    plt.xlim([-0.5, num_steps_plt-0.5])
    plt.ylabel('reward ' + ' (' + str(np.round(np.mean(perf), 2)) + ')')
    if states is not None:
        ax = plt.gca()
        ax.set_xticks([])
        plt.subplot(rows, 1, 4)
        plt.imshow(states[:num_steps_plt, int(states.shape[1]/2):].T,
                   aspect='auto')
        plt.title('network activity')
        plt.ylabel('neurons')
        ax = plt.gca()

    plt.xlabel('timesteps')
    plt.tight_layout()
    plt.show()
    if folder != '':
        f.savefig(folder + '/env_struct.png')
        plt.close(f)

    return f
