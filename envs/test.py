# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import sys

print(sys.argv[1])

if sys.argv[1] == 'mante':
    import mante

    env = mante.Mante(dt=0.2)
    print(env.action_space)

    state, rew, status = env.step(env.action_space.sample())

    print(state)
    print(status)
    print(rew)
elif sys.argv[1] == 'priors':
    import priors

    env = priors.Priors(dt=0.2)
    print(env.action_space)
    for stp in range(11):
        state, rew, status, info = env.step(env.action_space.sample())

        print(state)
        print(status)
        print(rew)
        print(info)
