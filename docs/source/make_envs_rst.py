from neurogym.envs import ALL_ENVS

envs = ALL_ENVS.keys()

string = """
neurogym.envs
===================================

"""

for env in ALL_ENVS.values():
    string += '.. autoclass:: ' + env.split(':')[0] + '.' + env.split(':')[1] + '\n'
    string += '    :members:\n'
    string += '    :exclude-members: new_trial\n\n'


with open('envs.rst', 'w') as f:
    f.write(string)