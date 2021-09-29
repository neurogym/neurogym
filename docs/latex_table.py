#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 15:36:04 2021

@author: molano
"""

"""Script to make environment md"""

import gym
import neurogym as ngym
from neurogym.utils.info import info, info_wrapper
from neurogym.envs.registration import ALL_ENVS
FOLDER = '/home/molano/Desktop/'
ref_list = ['munoz2004look', 'wang2018prefrontal', 'mante2013context',
            'daw2011model', 'barak2010neuronal', 'freedman2006experience',
            'miller1996neural', 'zhang2019active', 'rose2016reactivation',
            'padoa2006neurons', 'sarafyazd2019hierarchical', 'genovesio2009feature',
            'wang2018flexible', 'egger2019internal', 'britten1992analysis',
            'inagaki2019discrete', 'kiani2009representation',
            'yang2007probabilistic', 'scott2015sources',
            'georgopoulos1986neuronal', 'tadin2003perceptual']
SOURCE_ROOT = 'https://github.com/gyyang/neurogym/blob/master/'


def add_link(text, link):
    # Add link to a within document location
    return '[{:s}](#{:s})'.format(text, link)


def write_doc(write_type):
    if write_type == 'tasks':
        all_items = ngym.all_envs()
        info_fn = info
        fname = FOLDER+'envs_table.txt'

    elif write_type == 'wrappers':
        all_items = ngym.all_wrappers()
        info_fn = info_wrapper
        fname = FOLDER+'wrappers_table.txt'
    else:
        raise ValueError

    string = '\\begin{center}\n\\begin{tabular}' +\
        '{ | m{7cm}| m{9cm} |}\n\\hline\n'
    counter = 0
    for name in all_items:
        try:
            # Get information about individual task or wrapper
            info_string = info_fn(name)
            name = name.replace('-v0', '')
            info_string = info_string.replace('\n', ' ')
            info_string = info_string.replace('Doc: ', '')
            string += name+'~\\cite{bibreference} &'
            start = info_string.find(name)+len(name)
            start = 0 if start == -1 else start
            end = info_string.find('Args:')
            end = info_string.find('Reference paper') if end == -1 else end
            string += info_string[start:end]+'\\\\\n\\hline\n'
            counter += 1
            if (counter+1) % 5 == 0:
                string += '\\end{tabular}\n\\end{center}\n'
                string += '\\begin{center}\n\\begin{tabular}' +\
                    '{ | m{7cm}| m{9cm} |}\n\\hline\n'

        except BaseException as e:
            print('Failure in ', name)
            print(e)
    string += '\\end{tabular}\n\\end{center}\n'
    with open(fname, 'w') as f:
        f.write(string)
        print('------------')
        print(string)


def main():
    # write_doc('tasks')
    write_doc('wrappers')


if __name__ == '__main__':
    main()
