"""Plotting functions."""

import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import gym


# TODO: This is changing user's plotting behavior for non-neurogym plots
mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'


def plot_env(env, num_steps=200, num_trials=None, def_act=None, model=None,
             name=None, legend=True, ob_traces=[], fig_kwargs={}, fname=None):
    """Plot environment with agent.

    Args:
        env: already built neurogym task or name of it
        num_steps: number of steps to run the task
        num_trials: if not None, the number of trials to run
        def_act: if not None (and model=None), the task will be run with the
                 specified action
        model: if not None, the task will be run with the actions predicted by
               model, which so far is assumed to be created and trained with the
               stable-baselines toolbox:
                   (https://github.com/hill-a/stable-baselines)
        name: title to show on the rewards panel
        legend: whether to show the legend for actions panel or not.
        ob_traces: if != [] observations will be plot as traces, with the labels
                    specified by ob_traces
        fig_kwargs: figure properties admited by matplotlib.pyplot.subplots() fun.
        fname: if not None, save fig or movie to fname
    """
    # We don't use monitor here because:
    # 1) env could be already prewrapped with monitor
    # 2) monitor will save data and so the function will need a folder

    if isinstance(env, str):
        env = gym.make(env)
    if name is None:
        name = type(env).__name__
    data = run_env(env=env, num_steps=num_steps, num_trials=num_trials,
                   def_act=def_act, model=model)

    fig = fig_(
        data['ob'], data['actions'],
        gt=data['gt'], rewards=data['rewards'],
        legend=legend, performance=data['perf'],
        states=data['states'], name=name, ob_traces=ob_traces,
        fig_kwargs=fig_kwargs, env=env, fname=fname
    )

    return fig


def run_env(env, num_steps=200, num_trials=None, def_act=None, model=None):
    observations = []
    ob_cum = []
    state_mat = []
    rewards = []
    actions = []
    actions_end_of_trial = []
    gt = []
    perf = []
    ob = env.reset()  # TODO: not saving this first observation
    ob_cum_temp = ob

    if num_trials is not None:
        num_steps = 1e5  # Overwrite num_steps value

    trial_count = 0
    for stp in range(int(num_steps)):
        if model is not None:
            action, _states = model.predict(ob)
            if isinstance(action, float) or isinstance(action, int):
                action = [action]
            if len(_states) > 0:
                state_mat.append(_states)
        elif def_act is not None:
            action = def_act
        else:
            action = env.action_space.sample()
        ob, rew, done, info = env.step(action)
        ob_cum_temp += ob
        ob_cum.append(ob_cum_temp.copy())
        if isinstance(info, list):
            info = info[0]
            ob_aux = ob[0]
            rew = rew[0]
            done = done[0]
            action = action[0]
        else:
            ob_aux = ob

        if done:
            env.reset()
        observations.append(ob_aux)
        rewards.append(rew)
        actions.append(action)
        if 'gt' in info.keys():
            gt.append(info['gt'])
        else:
            gt.append(0)

        if info['new_trial']:
            actions_end_of_trial.append(action)
            perf.append(info['performance'])
            ob_cum_temp = np.zeros_like(ob_cum_temp)
            trial_count += 1
            if num_trials is not None and trial_count >= num_trials:
                break
        else:
            actions_end_of_trial.append(-1)
            perf.append(-1)

    if model is not None and len(state_mat) > 0:
        states = np.array(state_mat)
        states = states[:, 0, :]
    else:
        states = None

    data = {
        'ob': np.array(observations).astype(np.float),
        'ob_cum': np.array(ob_cum).astype(np.float),
        'rewards': rewards,
        'actions': actions,
        'perf': perf,
        'actions_end_of_trial': actions_end_of_trial,
        'gt': gt,
        'states': states
    }
    return data


# TODO: Change name, fig_ not a good name
def fig_(ob, actions, gt=None, rewards=None, performance=None, states=None,
         legend=True, ob_traces=None, name='', fname=None, fig_kwargs={},
         env=None):
    """Visualize a run in a simple environment.

    Args:
        ob: np array of observation (n_step, n_unit)
        actions: np array of action (n_step, n_unit)
        gt: np array of groud truth
        rewards: np array of rewards
        performance: np array of performance
        states: np array of network states
        name: title to show on the rewards panel and name to save figure
        fname: if != '', where to save the figure
        legend: whether to show the legend for actions panel or not.
        ob_traces: None or list.
            If list, observations will be plot as traces, with the labels
            specified by ob_traces
        fig_kwargs: figure properties admited by matplotlib.pyplot.subplots() fun.
        env: environment class for extra information
    """
    ob = np.array(ob)
    actions = np.array(actions)

    if len(ob.shape) == 2:
        return plot_env_1dbox(
            ob, actions, gt=gt, rewards=rewards,
            performance=performance, states=states, legend=legend,
            ob_traces=ob_traces, name=name, fname=fname,
            fig_kwargs=fig_kwargs, env=env
        )
    elif len(ob.shape) == 4:
        return plot_env_3dbox(
            ob, actions, fname=fname, env=env
        )
    else:
        raise ValueError('ob shape {} not supported'.format(str(ob.shape)))


def plot_env_1dbox(
        ob, actions, gt=None, rewards=None, performance=None, states=None,
        legend=True, ob_traces=None, name='', fname=None, fig_kwargs={},
        env=None):
    """Plot environment with 1-D Box observation space."""
    if len(ob.shape) != 2:
        raise ValueError('ob has to be 2-dimensional.')
    steps = np.arange(ob.shape[0])  # XXX: +1? 1st ob doesn't have action/gt

    n_row = 2  # observation and action
    n_row += rewards is not None
    n_row += performance is not None
    n_row += states is not None

    gt_colors = 'gkmcry'
    if not fig_kwargs:
        fig_kwargs = dict(sharex=True, figsize=(5, n_row*1.2))

    f, axes = plt.subplots(n_row, 1, **fig_kwargs)
    i_ax = 0
    # ob
    ax = axes[i_ax]
    i_ax += 1
    if ob_traces:
        assert len(ob_traces) == ob.shape[1],\
            'Please provide label for each of the '+str(ob.shape[1]) +\
            ' traces in the observations'
        yticks = []
        for ind_tr, tr in enumerate(ob_traces):
            ax.plot(ob[:, ind_tr], label=ob_traces[ind_tr])
            yticks.append(np.mean(ob[:, ind_tr]))
        if legend:
            ax.legend()
        ax.set_xlim([-0.5, len(steps)-0.5])
        ax.set_yticks(yticks)
        ax.set_yticklabels(ob_traces)
    else:
        ax.imshow(ob.T, aspect='auto', origin='lower')
        if env and hasattr(env.observation_space, 'name'):
            # Plot environment annotation
            yticks = []
            yticklabels = []
            for key, val in env.observation_space.name.items():
                yticks.append((np.min(val)+np.max(val))/2)
                yticklabels.append(key)
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
        else:
            ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if name:
        ax.set_title(name + ' env')
    ax.set_ylabel('Obs.')
    ax.set_xticks([])
    # actions
    ax = axes[i_ax]
    i_ax += 1
    if len(actions.shape) > 1:
        # Changes not implemented yet
        ax.plot(steps, actions, marker='+', label='Actions')
    else:
        ax.plot(steps, actions, marker='+', label='Actions')
    if gt is not None:
        gt = np.array(gt)
        if len(gt.shape) > 1:
            for ind_gt in range(gt.shape[1]):
                ax.plot(steps, gt[:, ind_gt], '--'+gt_colors[ind_gt],
                        label='Ground truth '+str(ind_gt))
        else:
            ax.plot(steps, gt, '--'+gt_colors[0], label='Ground truth')
    ax.set_xlim([-0.5, len(steps)-0.5])
    ax.set_ylabel('Act.')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if legend:
        ax.legend()
    if env and hasattr(env.action_space, 'name'):
        # Plot environment annotation
        yticks = []
        yticklabels = []
        for key, val in env.action_space.name.items():
            yticks.append((np.min(val) + np.max(val)) / 2)
            yticklabels.append(key)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
    if n_row > 2:
        ax.set_xticks([])
    # rewards
    if rewards is not None:
        ax = axes[i_ax]
        i_ax += 1
        ax.plot(steps, rewards, 'r', label='Rewards')
        ax.set_ylabel('Rew.')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if legend:
            ax.legend()
        ax.set_xlim([-0.5, len(steps)-0.5])

        if env and hasattr(env, 'rewards') and env.rewards:
            # Plot environment annotation
            yticks = []
            yticklabels = []
            for key, val in env.rewards.items():
                yticks.append(val)
                yticklabels.append('{:s} {:0.2f}'.format(key[:4], val))
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
    if n_row > 3:
        ax.set_xticks([])
    # performance
    if performance is not None:
        ax = axes[i_ax]
        i_ax += 1
        ax.plot(steps, performance, 'k', label='Performance')
        ax.set_ylabel('Performance')
        performance = np.array(performance)
        mean_perf = np.mean(performance[performance != -1])
        ax.set_title('Mean performance: ' + str(np.round(mean_perf, 2)))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if legend:
            ax.legend()
        ax.set_xlim([-0.5, len(steps)-0.5])

    # states
    if states is not None:
        ax.set_xticks([])
        ax = axes[i_ax]
        i_ax += 1
        plt.imshow(states[:, int(states.shape[1]/2):].T,
                   aspect='auto')
        ax.set_title('Activity')
        ax.set_ylabel('Neurons')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ax.set_xlabel('Steps')
    plt.tight_layout()
    if fname:
        fname = str(fname)
        if not (fname.endswith('.png') or fname.endswith('.svg')):
            fname += '.png'
        f.savefig(fname, dpi=300)
        plt.close(f)
    return f


def plot_env_3dbox(ob, actions=None, fname='', env=None):
    """Plot environment with 3-D Box observation space."""
    ob = ob.astype(np.uint8)  # TODO: Temporary
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.axis('off')
    im = ax.imshow(ob[0], animated=True)

    def animate(i, *args, **kwargs):
        im.set_array(ob[i])
        return im,

    if env is not None:
        interval = env.dt
    else:
        interval = 50
    ani = animation.FuncAnimation(fig, animate, frames=ob.shape[0],
                                  interval=interval)
    if fname:
        writer = animation.writers['ffmpeg'](fps=int(1000 / interval))
        fname = str(fname)
        if not fname.endswith('.mp4'):
            fname += '.mp4'
        ani.save(fname, writer=writer, dpi=300)


def plot_rew_across_training(folder, window=500, ax=None,
                             fkwargs={'c': 'tab:blue'}, ytitle='',
                             legend=False, zline=False, metric_name='reward'):
    data = put_together_files(folder)
    if data:
        sv_fig = False
        if ax is None:
            sv_fig = True
            f, ax = plt.subplots(figsize=(8, 8))
        metric = data[metric_name]
        if isinstance(window, float):
            if window < 1.0:
                window = int(metric.size * window)
        mean_metric = np.convolve(metric, np.ones((window,))/window,
                                  mode='valid')
        ax.plot(mean_metric, **fkwargs)  # add color, label etc.
        ax.set_xlabel('trials')
        if not ytitle:
            ax.set_ylabel('mean ' + metric_name + ' (running window' +
                          ' of {:d} trials)'.format(window))
        else:
            ax.set_ylabel(ytitle)
        if legend:
            ax.legend()
        if zline:
            ax.axhline(0, c='k', ls=':')
        if sv_fig:
            f.savefig(folder + '/mean_' + metric_name + '_across_training.png')
    else:
        print('No data in: ', folder)


def put_together_files(folder):
    files = glob.glob(folder + '/*_bhvr_data*npz')
    data = {}
    if len(files) > 0:
        files = order_by_sufix(files)
        file_data = np.load(files[0], allow_pickle=True)
        for key in file_data.keys():
            data[key] = file_data[key]

        for ind_f in range(1, len(files)):
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
    f = '/home/molano/res080220/SL_PerceptualDecisionMaking-v0_0/'
    plot_rew_across_training(folder=f)
