"""Script to make environment md"""

import gym
from neurogym import all_tasks
from neurogym.wrappers import all_wrappers
from neurogym.meta.tasks_info import info, info_wrapper

SOURCE_ROOT = 'https://github.com/gyyang/neurogym/blob/master/'
# TODO: Add link to source
def write_envs():
    md_file = 'envs.md'
    counter = 0
    string = ''
    names = ''
    for env_name in sorted(all_tasks.keys()):
        try:
            string += '---\n\n'
            string += info(env_name)
            source_link = all_tasks[env_name].split(':')[0].replace('.', '/')
            string += '[Source]({:s})\n\n'.format(SOURCE_ROOT+source_link+'.py')
            env = gym.make(env_name)
            # Adding automatic tagging using github's mechanism
            names += '[{:s}](#{:s})\n\n'.format(
                env_name, type(env).__name__.lower().replace(' ', '-'))
            counter += 1
        except BaseException as e:
            print('Failure in ', env_name)
            print(e)

    str1 = '### List of environments implemented\n\n'
    str2 = "* {0} tasks implemented so far.\n\n".format(counter)
    string = str1 + str2 + names + string
    with open(md_file, 'w') as f:
        f.write('* Under development, details subject to change\n\n')
        f.write(string)


def write_wrappers():
    string = ''
    names = ''
    counter = 0
    for wrapper_name in sorted(all_wrappers.keys()):
        try:
            string += info_wrapper(wrapper_name)
            names += '[{:s}](#{:s})\n\n'.format(
                wrapper_name, wrapper_name.lower().replace(' ', '-'))
            counter += 1
        except BaseException as e:
            print('Failure in ', wrapper_name)
            print(e)

    str1 = '### List of wrappers implemented\n\n'
    str2 = "* {0} wrappers implemented so far.\n\n".format(counter)
    string = str1 + str2 + names + string

    md_file = 'wrappers.md'
    with open(md_file, 'w') as f:
        f.write('* Under development, details subject to change\n\n')
        f.write(string)


def main():
    write_envs()
    write_wrappers()



if __name__ == '__main__':
    main()
