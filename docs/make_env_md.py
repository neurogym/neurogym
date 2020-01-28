"""Script to make environment md"""

from neurogym import all_tasks
from neurogym.wrappers import all_wrappers
from neurogym.meta.tasks_info import info, info_wrapper


def main():
    md_file = 'envs.md'
    counter = 0
    string1 = ''
    for env_name in sorted(all_tasks.keys()):
        try:
            string1 += info(env_name)
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
            string2 += info_wrapper(wrapper_name)
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
