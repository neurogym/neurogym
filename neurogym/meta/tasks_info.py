#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Formatting information about tasks and wrappers."""

import numpy as np
import matplotlib.pyplot as plt

import gym
import neurogym as ngym
from neurogym import all_tasks
from neurogym.wrappers import all_wrappers


METADATA_DEF_KEYS = ['description', 'paper_name', 'paper_link', 'timing',
                     'tags']


def info(task=None, show_code=False, show_fig=False, n_stps_plt=200, tags=[]):
    """Script to get tasks info"""
    if task is None:
        tasks = all_tasks.keys()
        string = ''
        counter = 0
        for env_name in sorted(tasks):
            env = gym.make(env_name)
            metadata = env.metadata
            if len(list(set(tags) - set(metadata['tags']))) == 0:
                string += env_name + '\n'
                counter += 1
        print('\n\n### List of environments implemented\n\n')
        print("* {0} tasks implemented so far.\n\n".format(counter))
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
            string += "Logic: {:s}\n\n".format(task_description)
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
            other_info = list(set(metadata.keys()) - set(METADATA_DEF_KEYS))
            if len(other_info) > 0:
                string += "Other parameters: \n\n"
                for key in other_info:
                    string += key + ' : ' + str(metadata[key]) + '\n\n'
            # tags
            tags = metadata['tags']
            string += 'Tags: '
            for tag in tags:
                string += tag + ', '
            string = string[:-2] + '.\n\n'

            # plot basic structure
            if show_fig:
                print('#### Example trials ####')
                plot_struct(env, n_stps_plt=n_stps_plt,
                            num_steps_env=n_stps_plt)
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
    return string


def info_wrapper(wrapper=None, show_code=False):
    """Script to get wrappers info"""
    if wrapper is None:
        wrappers = all_wrappers.keys()
        string = ''
        for env_name in sorted(wrappers):
            string += env_name + '\n'
        print('### List of wrappers implemented\n\n')
        print("* {0} wrappers implemented so far.\n\n".format(len(wrappers)))
        print(string)
    else:
        string = ''
        try:
            wrapp_ref = all_wrappers[wrapper]
            from_ = wrapp_ref[:wrapp_ref.find(':')]
            class_ = wrapp_ref[wrapp_ref.find(':')+1:]
            imported = getattr(__import__(from_, fromlist=[class_]), class_)
            metadata = imported.metadata
            string += "### {:s} wrapper ###\n\n".format(wrapper)
            paper_name = metadata.get('paper_name',
                                      None)
            paper_link = metadata.get('paper_link', None)
            wrapper_description = metadata.get('description',
                                               None) or 'Missing description'
            string += "Logic: {:s}\n\n".format(wrapper_description)
            if paper_name is not None:
                string += "Reference paper: \n\n"
                if paper_link is None:
                    string += "{:s}\n\n".format(paper_name)
                else:
                    string += "[{:s}]({:s})\n\n".format(paper_name,
                                                        paper_link)
            # add extra info
            other_info = list(set(metadata.keys()) - set(METADATA_DEF_KEYS))
            if len(other_info) > 0:
                string += "Input parameters: \n\n"
                for key in other_info:
                    string += key + ' : ' + str(metadata[key]) + '\n\n'

            # show source code
            if show_code:
                string += '''\n#### Source code #### \n\n'''
                import inspect
                lines = inspect.getsource(imported)
                string += lines + '\n\n'
        except BaseException as e:
            print('Failure in ', wrapper)
            print(e)
        print(string)
    return string


def plot_struct(env, num_steps_env=200, n_stps_plt=200,
                def_act=None, model=None, name=None, legend=True):
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
    fig_(obs, actions, gt, rewards, n_stps_plt, perf, legend=legend,
         states=states, name=name)
    data = {'obs': obs, 'obs_cum': obs_cum, 'rewards': rewards,
            'actions': actions, 'perf': perf,
            'actions_end_of_trial': actions_end_of_trial, 'gt': gt,
            'states': states}
    return data


def fig_(obs, actions, gt, rewards, n_stps_plt, perf, legend=True,
         obs_cum=None, states=None, name=''):
    if states is not None:
        rows = 4
    else:
        rows = 3

    f = plt.figure(figsize=(8, 8))
    # obs
    plt.subplot(rows, 1, 1)
    plt.imshow(obs[:n_stps_plt, :].T, aspect='auto')
    plt.title('observations ' + name + ' task')
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    # actions
    plt.subplot(rows, 1, 2)
    plt.plot(np.arange(n_stps_plt) + 0.,
             actions[:n_stps_plt], marker='+', label='actions')
    gt = np.array(gt)
    if len(gt.shape) == 2:
        gt = np.argmax(gt, axis=1)
    plt.plot(np.arange(n_stps_plt) + 0.,
             gt[:n_stps_plt], 'r', label='ground truth')
    plt.ylabel('actions')
    if legend:
        plt.legend()
    plt.xlim([-0.5, n_stps_plt-0.5])
    ax = plt.gca()
    ax.set_xticks([])
    # rewards
    plt.subplot(rows, 1, 3)
    plt.plot(np.arange(n_stps_plt) + 0.,
             rewards[:n_stps_plt], 'r')
    plt.xlim([-0.5, n_stps_plt-0.5])
    plt.ylabel('reward ' + ' (' + str(np.round(np.mean(perf), 2)) + ')')
    if states is not None:
        ax = plt.gca()
        ax.set_xticks([])
        plt.subplot(rows, 1, 4)
        plt.imshow(states[:n_stps_plt, int(states.shape[1]/2):].T,
                   aspect='auto')
        plt.title('network activity')
        plt.ylabel('neurons')
        ax = plt.gca()

    plt.xlabel('timesteps')
    plt.tight_layout()
    plt.show()
    return f


def get_all_tags(verbose=0):
    """Script to get all tags"""
    tasks = all_tasks.keys()
    tags = []
    for env_name in sorted(tasks):
        try:
            env = gym.make(env_name)
            metadata = env.metadata
            new_tags = list(set(metadata['tags']) - set(tags))
            tags += new_tags
        except BaseException as e:
            print('Failure in ', env_name)
            print(e)
    if verbose:
        print('\nTAGS:\n')
        for tag in tags:
            print(tag)
    return tags


if __name__ == '__main__':
    # get_all_tags(verbose=1)
    # info(tags=['supervised setting', 'n-alternative'])
    info('ChangingEnvironment-v0')
    # info('RDM-v0', show_code=True, show_fig=True)
#    info_wrapper()
#    info_wrapper('ReactionTime-v0', show_code=True)
