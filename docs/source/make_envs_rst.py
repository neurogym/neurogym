import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import gym
import neurogym as ngym
from neurogym.envs import ALL_ENVS
from neurogym.wrappers import ALL_WRAPPERS


def make_env_images():
    envs = ngym.all_envs()
    for env_name in envs:
        env = gym.make(env_name, **{'dt': 20})
        action = np.zeros_like(env.action_space.sample())
        fname = Path(__file__).parent / 'images' / (env_name + '_examplerun.png')
        # fname = os.path.join('.', 'images', env_name + '_examplerun')
        ngym.utils.plot_env(env, num_trials=2, def_act=action, fname=fname)
        plt.close()


def make_envs():
    string = 'Environments\n'
    string += '===================================\n\n'

    for key, val in sorted(ALL_ENVS.items()):
        string += key + '\n'+'-'*50+'\n'
        string += '.. autoclass:: ' + val.split(':')[0] + '.' + val.split(':')[1] + '\n'
        string += '    :members:\n'
        string += '    :exclude-members: new_trial\n\n'

        env = gym.make(key)
        # Add paper
        paper_name = env.metadata.get('paper_name', '')
        paper_link = env.metadata.get('paper_link', '')
        if paper_name:
            string += '    Reference paper\n'
            paper_name = paper_name.replace('\n', ' ')
            string += '        `{:s} <{:s}>`__\n\n'.format(paper_name, paper_link)
            # string += '    .. __{:s}:\n        {:s}\n\n'.format(paper_name, paper_link)

        # Add tags
        string += '    Tags\n'
        for tag in env.metadata.get('tags', []):
            string += '        :ref:`tag-{:s}`, '.format(tag)
        string = string[:-2]
        string += '\n\n'

        # Add image
        string += '    Sample run\n'
        image_path = Path('images') / (key + '_examplerun.png')
        if (Path(__file__).parent / image_path).exists():
            string += ' '*8 + '.. image:: {:s}\n'.format(str(image_path))
            string += ' '*12 + ':width: 600\n\n'

    with open(Path(__file__).parent / 'envs.rst', 'w') as f:
        f.write(string)


def make_tags():
    string = 'Tags\n'
    string += '===================================\n\n'

    all_tags = ngym.all_tags()

    for tag in sorted(all_tags):
        string += '.. _tag-{:s}:\n\n'.format(tag)
        string += tag + '\n--------------------------------\n'
        for env in ngym.all_envs(tag=tag):
            string += '    :class:`{:s} <{:s}>`\n'.format(env, ALL_ENVS[
                env].replace(':', '.'))
        string += '\n'
    with open(Path(__file__).parent / 'tags.rst', 'w') as f:
        f.write(string)

    string = 'Wrappers\n'
    string += '===================================\n\n'

    for key, val in ALL_WRAPPERS.items():
        string += key + '\n' + '-' * 50 + '\n'
        string += '.. autoclass:: ' + val.split(':')[0] + '.' + val.split(':')[
            1] + '\n'
        string += '    :members:\n'
        string += '    :exclude-members: new_trial\n\n'

    with open(Path(__file__).parent / 'wrappers.rst', 'w') as f:
        f.write(string)


def main():
    # make_env_images()
    make_envs()
    make_tags()


if __name__ == '__main__':
    main()
