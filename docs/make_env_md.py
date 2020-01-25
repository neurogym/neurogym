"""Script to make environment md"""


import gym
import neurogym as ngym
from neurogym import all_tasks
from neurogym.wrappers import all_wrappers


def main():
    md_file = 'envs.md'
    counter = 0
    string1 = ''
    for env_name in sorted(all_tasks.keys()):
        try:
            env = gym.make(env_name)
            metadata = env.metadata

            string1 += "#### {:s}\n\n".format(type(env).__name__)

            paper_name = metadata.get('paper_name',
                                      None) or 'Missing paper name'
            paper_link = metadata.get('paper_link', None)
            task_description = metadata.get('description',
                                            None) or 'Missing description'
            string1 += "{:s}\n\n".format(task_description)
            string1 += "Reference paper: \n\n"
            if paper_link is None:
                string1 += "{:s}\n\n".format(paper_name)
                string1 += 'Missing paper link\n\n'
            else:
                string1 += "[{:s}]({:s})\n\n".format(paper_name, paper_link)

            if isinstance(env, ngym.EpochEnv):
                timing = metadata['timing']
                string1 += 'Default Epoch timing (ms) \n\n'
                for key, val in timing.items():
                    dist, args = val
                    string1 += key + ' : ' + dist + ' ' + str(args) + '\n\n'
            counter += 1
        except BaseException as e:
            print('Failure in ', env_name)
            print(e)

    str1 = '### List of environments implemented\n\n'
    str2 = "* {0} tasks implemented so far.\n\n".format(counter)
    string1 = str1 + str2 + string1

    string2 = ''
    counter = 0
    for wrapper_name in sorted(all_wrappers.keys()):
        try:
            wrapp_ref = all_wrappers[wrapper_name]
            from_ = wrapp_ref[:wrapp_ref.find(':')]
            class_ = wrapp_ref[wrapp_ref.find(':')+1:]
            imported = getattr(__import__(from_, fromlist=[class_]), class_)
            metadata = imported.metadata

            string2 += "#### {:s}\n\n".format(wrapper_name)

            paper_name = metadata.get('paper_name',
                                      None)
            paper_link = metadata.get('paper_link', None)
            task_description = metadata.get('description',
                                            None) or 'Missing description'
            string2 += "{:s}\n\n".format(task_description)
            if paper_name is not None:
                string2 += "Reference paper: \n\n"
                if paper_link is None:
                    string2 += "{:s}\n\n".format(paper_name)
                else:
                    string2 += "[{:s}]({:s})\n\n".format(paper_name,
                                                         paper_link)

            counter += 1
        except BaseException as e:
            print('Failure in ', wrapper_name)
            print(e)

    str1 = '### List of wrappers implemented\n\n'
    str2 = "* {0} wrappers implemented so far.\n\n".format(counter)
    string2 = str1 + str2 + string2

    with open(md_file, 'w') as f:
        f.write('* Under development, details subject to change\n\n')
        f.write(string1)
        f.write('* \n\n\n\n\n')
        f.write(string2)


if __name__ == '__main__':
    main()
