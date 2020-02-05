"""Plotting functions."""

import numpy as np
import matplotlib.pyplot as plt
import glob
import gym


def plot_env(env, num_steps_env=200,
             def_act=None, model=None, name=None, legend=True, fig_kwargs={}):
    """
    env: already built neurogym task or name of it
    num_steps_env: number of steps to run the task
    def_act: if not None (and model=None), the task will be run with the
             specified action
    model: if not None, the task will be run with the actions predicted by
           model, which so far is assumed to be created and trained with the
           stable-baselines toolbox:
               (https://github.com/hill-a/stable-baselines)
    name: title to show on the rewards panel
    legend: whether to show the legend for actions panel or not.
    """
    # TODO: Can't we use Monitor here? We could but:
    # 1) env could be already prewrapped with monitor
    # 2) monitor will save data and so the function will need a folder
    if isinstance(env, str):
        env = gym.make(env)
    if name is None:
        name = type(env).__name__
    observations, obs_cum, rewards, actions, perf, actions_end_of_trial,\
        gt, states = run_env(env=env, num_steps_env=num_steps_env,
                             def_act=def_act, model=model)
    obs_cum = np.array(obs_cum)
    obs = np.array(observations)
    fig_(obs, actions, gt, rewards, legend=legend,
         states=states, name=name, fig_kwargs=fig_kwargs)
    data = {'obs': obs, 'obs_cum': obs_cum, 'rewards': rewards,
            'actions': actions, 'perf': perf,
            'actions_end_of_trial': actions_end_of_trial, 'gt': gt,
            'states': states}
    return data


def run_env(env, num_steps_env=200, def_act=None, model=None):
    observations = []
    obs_cum = []
    state_mat = []
    rewards = []
    actions = []
    actions_end_of_trial = []
    gt = []
    perf = []
    obs = env.reset()  # TODO: not saving this first observation
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
        if 'gt' in info.keys():
            gt.append(info['gt'])
        else:
            gt.append(0)
    if model is not None:
        states = np.array(state_mat)
        states = states[:, 0, :]
    else:
        states = None
    return observations, obs_cum, rewards, actions, perf,\
        actions_end_of_trial, gt, states


def fig_(obs, actions, gt=None, rewards=None, states=None,
         legend=True, name='', folder='', fig_kwargs={}):
    if len(obs.shape) != 2:
        raise ValueError('obs has to be 2-dimensional.')
    # TODO: Add documentation
    steps = np.arange(obs.shape[0])  # XXX: +1? 1st obs doesn't have action/gt

    n_row = 2  # observation and action
    n_row += rewards is not None
    n_row += states is not None

    gt_colors = 'gkmcry'
    if not fig_kwargs:
        fig_kwargs=dict(sharex=True, figsize=(5, n_row*1.5))

    f, axes = plt.subplots(n_row, 1, **fig_kwargs)
    # obs
    ax = axes[0]
    ax.imshow(obs.T, aspect='auto')
    if name:
        ax.set_title(name + ' env')
    ax.set_ylabel('Observations')
    ax.set_yticks([])
    ax.set_xlim([-0.5, len(steps)-0.5])

    # actions
    ax = axes[1]
    ax.plot(steps, actions, marker='+', label='Actions')

    if gt is not None:
        gt = np.array(gt)
        if len(gt.shape) > 1:
            for ind_gt in range(gt.shape[1]):
                ax.plot(steps, gt[:, ind_gt], '--'+gt_colors[ind_gt],
                        label='Ground truth '+str(ind_gt))
        else:
            ax.plot(steps, gt, '--'+gt_colors[0], label='Ground truth')

    ax.set_ylabel('Actions')
    if legend:
        ax.legend()

    if rewards is not None:
        # rewards
        ax = axes[2]
        ax.plot(steps, rewards, 'r')
        ax.set_ylabel('Reward')

    if states is not None:
        ax.set_xticks([])
        ax = axes[3]
        plt.imshow(states[:, int(states.shape[1]/2):].T,
                   aspect='auto')
        ax.set_title('Activity')
        ax.set_ylabel('Neurons')

    ax.set_xlabel('Steps')
    plt.tight_layout()
    if folder is not None and folder != '':
        f.savefig(folder + '/env_struct.png')
        plt.close(f)

    return f


def plot_rew_across_training(folder, window=500):
    data = put_together_files(folder)
    f = plt.figure(figsize=(8, 8))
    reward = data['reward']
    mean_reward = np.convolve(reward, np.ones((window,))/window, mode='valid')
    plt.plot(mean_reward)
    plt.xlabel('trials')
    plt.ylabel('mean reward (running window of {:d} trials'.format(window))
    f.savefig(folder + '/mean_reward_across_training.png')


def put_together_files(folder):
    files = glob.glob(folder + '/*_bhvr_data*npz')
    files = order_by_sufix(files)
    file_data = np.load(files[0], allow_pickle=True)
    data = {}
    for key in file_data.keys():
        data[key] = file_data[key]

    for ind_f in range(len(files)):
        file_data = np.load(files[ind_f], allow_pickle=True)
        for key in file_data.keys():
            data[key] = np.concatenate((data[key], file_data[key]))
    np.savez(folder + '/bhvr_data_all.npz', **data)
    return data


def order_by_sufix(file_list):
    sfx = [int(x[x.rfind('_')+1:x.rfind('.')]) for x in file_list]
    sorted_list = [x for _, x in sorted(zip(sfx, file_list))]
    return sorted_list


if __name__ == '__main__':
    f = '/home/molano/ngym_usage/results/dpa_tests/'
    plot_rew_across_training(folder=f)
