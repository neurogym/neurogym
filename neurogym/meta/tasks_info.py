#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:04:58 2020

@author: manuel
"""

import gym
import neurogym as ngym
from neurogym import all_tasks


def info(task=None):
    """Script to get tasks info"""
    if task is None:
        string = ''
        for env_name in sorted(all_tasks.keys()):
            string += env_name
        print('### List of environments implemented\n\n')
        print("* {0} tasks implemented so far.\n\n".format(len(all_tasks.keys())))
        print('* Under development, details subject to change\n\n')
        print(string)
    else:
        string = ''
        try:
            env = gym.make(task)
            metadata = env.metadata
            string += "#### {:s}\n\n".format(type(env).__name__)
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

            if isinstance(env, ngym.EpochEnv):
                timing = metadata['default_timing']
                string += 'Default Epoch timing (ms) \n\n'
                for key, val in timing.items():
                    dist, args = val
                    string += key + ' : ' + dist + ' ' + str(args) + '\n\n'
        except BaseException as e:
            print('Failure in ', env_name)
            print(e)


if __name__ == '__main__':
    info()
