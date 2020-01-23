#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:04:58 2020

@author: manuel
"""

"""Script to get tasks info"""


import gym
import neurogym as ngym
from neurogym import all_tasks


def info():

    counter = 0
    string = ''
    for env_name in sorted(all_tasks.keys()):
        try:
            env = gym.make(env_name)
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
            counter += 1
        except BaseException as e:
            print('Failure in ', env_name)
            print(e)

    print('### List of environments implemented\n\n')
    print("* {0} tasks implemented so far.\n\n".format(counter))
    print('* Under development, details subject to change\n\n')
    print(string)


if __name__ == '__main__':
    info()
