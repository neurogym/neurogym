import os

import gym
import neurogym as ngym
from neurogym.envs import ALL_ENVS
from neurogym.wrappers import ALL_WRAPPERS


def main():
    string = 'Environments\n'
    string += '===================================\n\n'

    for key, val in ALL_ENVS.items():
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
        image_path = os.path.join('images', key+'_examplerun.png')
        if os.path.isfile(image_path):
            string += '.. image:: {:s}\n    :width: 600\n\n'.format(image_path)

    with open('envs.rst', 'w') as f:
        f.write(string)

    string = 'Tags\n'
    string += '===================================\n\n'

    all_tags = ngym.all_tags()

    for tag in sorted(all_tags):
        string += '.. _tag-{:s}:\n\n'.format(tag)
        string += tag + '\n--------------------------------\n'
        for env in ngym.all_envs(tag=tag):
            string += '    :class:`{:s} <{:s}>`\n'.format(env, ALL_ENVS[env].replace(':', '.'))
        string += '\n'
    with open('tags.rst', 'w') as f:
        f.write(string)


    string = 'Wrappers\n'
    string += '===================================\n\n'

    for key, val in ALL_WRAPPERS.items():
        string += key + '\n' + '-' * 50 + '\n'
        string += '.. autoclass:: ' + val.split(':')[0] + '.' + val.split(':')[1] + '\n'
        string += '    :members:\n'
        string += '    :exclude-members: new_trial\n\n'

    with open('wrappers.rst', 'w') as f:
        f.write(string)


if __name__ == '__main__':
    main()