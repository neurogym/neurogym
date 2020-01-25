"""Script to make environment md"""


import gym
import neurogym as ngym
from neurogym import all_tasks
from neurogym.wrappers import all_wrappers


def main():
    md_file = 'envs.md'
    counter = 0
    string = ''
    for env_name in sorted(all_tasks.keys()):
        try:
            env = gym.make(env_name)
            metadata = env.metadata

            string += "#### {:s}\n\n".format(type(env).__name__)

            paper_name = metadata.get('paper_name',
                                      None) or 'Missing paper name'
            paper_link = metadata.get('paper_link', None)
            task_description = metadata.get('description',
                                            None) or 'Missing description'
            string += "{:s}\n\n".format(task_description)
            string += "Reference paper: \n\n"
            if paper_link is None:
                string += "{:s}\n\n".format(paper_name)
                string += 'Missing paper link\n\n'
            else:
                string += "[{:s}]({:s})\n\n".format(paper_name, paper_link)

            if isinstance(env, ngym.EpochEnv):
                timing = metadata['timing']
                string += 'Default Epoch timing (ms) \n\n'
                for key, val in timing.items():
                    dist, args = val
                    string += key + ' : ' + dist + ' ' + str(args) + '\n\n'
            counter += 1
        except BaseException as e:
            print('Failure in ', env_name)
            print(e)

    with open(md_file, 'w') as f:
        f.write('### List of environments implemented\n\n')
        f.write("* {0} tasks implemented so far.\n\n".format(counter))
        f.write(string)

    counter = 0
    string = ''
    for wrapper_name in sorted(all_wrappers.keys()):
#        try:
        wrapp_ref = all_wrappers[wrapper_name]
        from_ = wrapp_ref[:wrapp_ref.find(':')]
        class_ = wrapp_ref[wrapp_ref.find(':')+1:]
        imported = getattr(__import__(from_, fromlist=[class_]), class_)
        metadata = imported.metadata

        string += "#### {:s}\n\n".format(type(env).__name__)

        paper_name = metadata.get('paper_name',
                                  None) or 'Missing paper name'
        paper_link = metadata.get('paper_link', None)
        task_description = metadata.get('description',
                                        None) or 'Missing description'
        string += "{:s}\n\n".format(task_description)
        string += "Reference paper: \n\n"
        if paper_link is None:
            string += "{:s}\n\n".format(paper_name)
            string += 'Missing paper link\n\n'
        else:
            string += "[{:s}]({:s})\n\n".format(paper_name, paper_link)

        if isinstance(env, ngym.EpochEnv):
            timing = metadata['timing']
            string += 'Default Epoch timing (ms) \n\n'
            for key, val in timing.items():
                dist, args = val
                string += key + ' : ' + dist + ' ' + str(args) + '\n\n'
        counter += 1
#        except BaseException as e:
#            print('Failure in ', env_name)
#            print(e)

    with open(md_file, 'w') as f:
        f.write('### List of wrappers implemented\n\n')
        f.write("* {0} wrappers implemented so far.\n\n".format(counter))
        f.write('* Under development, details subject to change\n\n')
        f.write(string)



if __name__ == '__main__':
    main()
