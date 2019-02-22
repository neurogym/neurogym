# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import mante

env = mante.Mante(dt=0.2)
print(env.action_space)

state, rew, status = env.step(env.action_space.sample())

print(state)
print(status)
print(rew)