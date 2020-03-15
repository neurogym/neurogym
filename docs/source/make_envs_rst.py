import gym
import neurogym as ngym
from neurogym.envs import ALL_ENVS

envs = ALL_ENVS.keys()

string = """
Environments
===================================

"""

for key, val in ALL_ENVS.items():
    string += '.. autoclass:: ' + val.split(':')[0] + '.' + val.split(':')[1] + '\n'
    string += '    :members:\n'
    string += '    :exclude-members: new_trial\n\n'

    string += '    Tags\n'
    env = gym.make(key)
    for tag in env.metadata.get('tags', []):
        string += '        :ref:`tag-{:s}`, '.format(tag)
    string = string[:-2]
    string += '\n\n'


with open('envs.rst', 'w') as f:
    f.write(string)

string = """
Tags
===================================

"""

all_tags = ngym.all_tags()

for tag in sorted(all_tags):
    string += '.. _tag-{:s}:\n\n'.format(tag)
    string += tag + '\n--------------------------------\n'
    for env in ngym.all_envs(tag=tag):
        string += '    :class:`{:s} <{:s}>`\n'.format(env, ALL_ENVS[env].replace(':', '.'))
    string += '\n'
with open('tags.rst', 'w') as f:
    f.write(string)