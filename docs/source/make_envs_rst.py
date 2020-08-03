import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import gym
import neurogym as ngym
from neurogym.envs import ALL_ENVS
from neurogym.wrappers import ALL_WRAPPERS


ENV_IGNORE = ['Nothing-v0', 'Pneumostomeopening-v0']
all_envs = dict()
for key, val in sorted(ALL_ENVS.items()):
    if key in ENV_IGNORE:
        continue
    all_envs[key] = val


def make_env_images():
    envs = all_envs.keys()
    for env_name in envs:
        env = gym.make(env_name, **{'dt': 20})
        action = np.zeros_like(env.action_space.sample())
        fname = Path(__file__).parent / '_static' / (env_name + '_examplerun')
        ngym.utils.plot_env(env, num_trials=2, def_act=action, fname=fname)
        plt.close()


def make_envs():
    # Make envs/index.rst
    string = 'Environments\n'
    string += '===================================\n\n'
    string += '.. toctree::\n'
    string += '    :maxdepth: 1\n\n'
    for key, val in all_envs.items():
        string += ' ' * 4 + '{:s}\n'.format(key)
    with open(Path(__file__).parent / 'envs' / 'index.rst', 'w') as f:
        f.write(string)

    for key, val in all_envs.items():
        string = ''
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
        image_path = Path('_static') / (key + '_examplerun.tmp')

        suffix = None
        _image_path = (Path(__file__).parent / image_path)
        for s in ['.png', '.mp4']:  # Check suffix
            if _image_path.with_suffix(s).exists():
                suffix = s
                break

        if suffix is not None:
            image_path = str(image_path.with_suffix(suffix))
            if suffix == '.png':
                string += ' '*8 + '.. image:: ../{:s}\n'.format(image_path)
                string += ' ' * 12 + ':width: 600\n\n'
            elif suffix == '.mp4':
                string += ' ' * 8 + '.. video:: ../{:s}\n'.format(image_path)
                string += ' ' * 12 + ':width: 300\n'
                string += ' ' * 12 + ':height: 300\n'
                # string += ' ' * 12 + ':autoplay:\n'
                string += ' ' * 12 + ':loop:\n'

        with open(Path(__file__).parent / 'envs' / (key + '.rst'), 'w') as f:
            f.write(string)


def make_tags():
    string = 'Tags\n'
    string += '===================================\n\n'

    all_tags = ngym.all_tags()

    for tag in sorted(all_tags):
        string += '.. _tag-{:s}:\n\n'.format(tag)
        string += tag + '\n--------------------------------\n'
        for env in ngym.all_envs(tag=tag):
            if env in ENV_IGNORE:
                continue
            string += '    :class:`{:s} <{:s}>`\n'.format(env, all_envs[
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
    make_env_images()
    make_envs()
    make_tags()


if __name__ == '__main__':
    main()
