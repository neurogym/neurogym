#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:04:58 2020

@author: manuel
"""

import gym
import neurogym as ngym
from neurogym import all_tasks
import matplotlib.pyplot as plt
import numpy as np
metadata_basic_info = ['description', 'paper_name', 'paper_link', 'timing']


def info(task=None, show_code=False, n_stps_plt=200):
    """Script to get tasks info"""
    if task is None:
        tasks = all_tasks.keys()
        string = ''
        for env_name in sorted(tasks):
            string += env_name + '\n'
        print('### List of environments implemented\n\n')
        print("* {0} tasks implemented so far.\n\n".format(len(tasks)))
        print('* Under development, details subject to change\n\n')
        print(string)
    else:
        string = ''
        try:
            env = gym.make(task)
            metadata = env.metadata
            string += "### {:s} task ###\n\n".format(type(env).__name__)
            paper_name = metadata.get('paper_name',
                                      None) or 'Missing paper name'
            paper_link = metadata.get('paper_link', None)
            task_description = metadata.get('description',
                                            None) or 'Missing description'
            string += "{:s}\n\n".format(task_description)
            string += "Reference paper: \n\n"
            if paper_link is None:
                string += "{:s}\n\n".format(paper_name)
                string += 'Missing paper link\n\n'
            else:
                string += "[{:s}]({:s})\n\n".format(paper_name, paper_link)
            # add timing info
            if isinstance(env, ngym.EpochEnv):
                timing = metadata['timing']
                string += 'Default Epoch timing (ms) \n\n'
                for key, val in timing.items():
                    dist, args = val
                    string += key + ' : ' + dist + ' ' + str(args) + '\n\n'
            # add extra info
            other_info = list(set(metadata.keys()) - set(metadata_basic_info))
            for key in other_info:
                string += key + ' : ' + str(metadata[key]) + '\n\n'
            # plot basic structure
            print('#### Example trials ####')
            plot_struct(env, n_stps_plt=n_stps_plt, num_steps_env=n_stps_plt)
            # show source code
            if show_code:
                string += '''\n#### Source code #### \n\n'''
                import inspect
                task_ref = all_tasks[task]
                from_ = task_ref[:task_ref.find(':')]
                class_ = task_ref[task_ref.find(':')+1:]
                imported = getattr(__import__(from_, fromlist=[class_]),
                                   class_)
                lines = inspect.getsource(imported)
                string += lines + '\n\n'
        except BaseException as e:
            print('Failure in ', type(env).__name__)
            print(e)
        print(string)


def plot_struct(env, num_steps_env=200, n_stps_plt=200,
                def_act=None, model=None):
    if isinstance(env, str):
        env = gym.make(env)
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
            print(info)
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
        rows = 4
        states = np.array(state_mat)
        states = states[:, 0, :]
    else:
        rows = 3
        states = None
    obs = np.array(observations)
    obs_cum = np.array(obs_cum)
    plt.figure(figsize=(8, 8))
    plt.subplot(rows, 1, 1)
    plt.imshow(obs[:n_stps_plt, :].T, aspect='auto')
    plt.title('observations ' + type(env).__name__ + ' task')
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.subplot(rows, 1, 2)
    plt.plot(actions[:n_stps_plt], marker='+', label='actions')
    gt = np.array(gt)
    if len(gt.shape) == 2:
        gt = np.argmax(gt, axis=1)
    plt.plot(gt[:n_stps_plt], 'r', label='ground truth')
    plt.title('actions')
    plt.xlim([-0.5, n_stps_plt+0.5])
    plt.legend()
    ax = plt.gca()
    ax.set_xticks([])
    plt.subplot(rows, 1, 3)
    plt.plot(rewards[:n_stps_plt], 'r')
    plt.xlim([-0.5, n_stps_plt+0.5])
    plt.title('reward ' + ' (' + str(np.round(np.mean(perf), 3)) + ')')
    plt.xlabel('timesteps')
    plt.tight_layout()
    if model is not None:
        plt.subplot(rows, 1, 4)
        plt.imshow(states[:n_stps_plt, int(states.shape[1]/2):].T,
                   aspect='auto')
    plt.show()

    data = {'obs': obs, 'obs_cum': obs_cum, 'rewards': rewards,
            'actions': actions, 'perf': perf,
            'actions_end_of_trial': actions_end_of_trial, 'gt': gt,
            'states': states}
    return data


if __name__ == '__main__':
    info('RDM-v0', show_code=True)
