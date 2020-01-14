"""Script to make environment md"""


import json
import gym
import neurogym as ngym
from neurogym import all_tasks
from neurogym.ops import tasktools


def main():
    md_file = 'envs.md'

    with open(md_file, 'w') as f:
        for env_name in sorted(all_tasks.keys()):
            try:
                env = gym.make(env_name)
                metadata = env.metadata

                f.write("### {:s}\n\n".format(type(env).__name__))

                paper_name = metadata.get('paper_name', None) or 'Missing paper name'
                f.write("Original paper: \n\n{:s}\n\n".format(paper_name))

                paper_link = metadata.get('paper_link', None) or 'Missing paper link'
                f.write("Link: {:s}\n\n".format(paper_link))

                if isinstance(env, ngym.EpochEnv):
                    timing = metadata['default_timing']
                    string = 'Epochs \n\n'
                    for key, val in timing.items():
                        dist, args = val
                        string += key + ' : ' + dist + ' ' + str(args) + '\n\n'

                    f.write(string)

            except BaseException as e:
                print('Failure')
                print(e)


if __name__ == '__main__':
    main()