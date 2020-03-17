#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Formatting information about envs and wrappers."""

import inspect

import gym
from neurogym.core import env_string, METADATA_DEF_KEYS
from neurogym.envs import all_envs, ALL_ENVS
from neurogym.wrappers import ALL_WRAPPERS


def all_tasks():
    for task in sorted(ALL_ENVS):
        print(task)


def all_wrappers():
    for wrapper in sorted(ALL_WRAPPERS):
        print(wrapper)


def info(env=None, show_code=False):
    """Script to get envs info"""
    string = ''
    try:
        env_name = env
        env = gym.make(env)
        string = env_string(env)
        # show source code
        if show_code:
            string += '''\n#### Source code #### \n\n'''
            env_ref = ALL_ENVS[env_name]
            from_, class_ = env_ref.split(':')
            imported = getattr(__import__(from_, fromlist=[class_]), class_)
            lines = inspect.getsource(imported)
            string += lines + '\n\n'
        print(string)
    except BaseException as e:
        print('Failure in ', type(env).__name__)
        print(e)
    return string


def info_wrapper(wrapper=None, show_code=False):
    """Script to get wrappers info"""
    string = ''
    try:
        wrapp_ref = ALL_WRAPPERS[wrapper]
        from_, class_ = wrapp_ref.split(':')
        imported = getattr(__import__(from_, fromlist=[class_]), class_)
        metadata = imported.metadata
        string += "### {:s}\n\n".format(wrapper)
        paper_name = metadata.get('paper_name',
                                  None)
        paper_link = metadata.get('paper_link', None)
        wrapper_description = metadata.get('description',
                                           None) or 'Missing description'
        string += "Logic: {:s}\n\n".format(wrapper_description)
        if paper_name is not None:
            string += "Reference paper \n\n"
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
            lines = inspect.getsource(imported)
            string += lines + '\n\n'
    except BaseException as e:
        print('Failure in ', wrapper)
        print(e)
    print(string)
    return string


def all_tags(verbose=0):
    """Script to get all tags"""
    envs = all_envs()
    tags = []
    for env_name in sorted(envs):
        try:
            env = gym.make(env_name)
            metadata = env.metadata
            tags += metadata.get('tags', [])
        except BaseException as e:
            print('Failure in ', env_name)
            print(e)
    tags = set(tags)
    if verbose:
        print('\nTAGS:\n')
        for tag in tags:
            print(tag)
    return tags


if __name__ == '__main__':
    all_tasks()
    # get_all_tags(verbose=1)
    # info(tags=['supervised', 'n-alternative'])
    # info('PerceptualDecisionMaking-v0', show_code=True)
    # info('PerceptualDecisionMaking-v0', show_code=True, show_fig=True)
#    info_wrapper('ReactionTime-v0', show_code=True)
