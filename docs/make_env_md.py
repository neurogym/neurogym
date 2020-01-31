"""Script to make environment md"""

import gym
from neurogym import all_tasks
from neurogym.wrappers import all_wrappers
from neurogym.meta.tasks_info import info, info_wrapper


SOURCE_ROOT = 'https://github.com/gyyang/neurogym/blob/master/'


def write_doc(write_type):
    if write_type == 'tasks':
        all_items = all_tasks
        info_fn = info
        fname = 'envs.md'
    elif write_type == 'wrappers':
        all_items = all_wrappers
        info_fn = info_wrapper
        fname = 'wrappers.md'
    else:
        raise ValueError

    string = ''
    names = ''
    counter = 0

    for name in sorted(all_items.keys()):
        try:
            string += info_fn(name)
            names += '[{:s}](#{:s})\n\n'.format(
                name, name.lower().replace(' ', '-'))
            source_link = all_items[name].split(':')[0].replace('.', '/')
            string += '[Source]({:s})\n\n'.format(
                SOURCE_ROOT + source_link + '.py')
            counter += 1
        except BaseException as e:
            print('Failure in ', name)
            print(e)

    str1 = '### List of {:s} implemented\n\n'.format(write_type)
    str2 = '* {:d} {:s} implemented so far.\n\n'.format(counter, write_type)
    string = str1 + str2 + names + string

    with open(fname, 'w') as f:
        f.write('* Under development, details subject to change\n\n')
        f.write(string)


def main():
    write_doc('tasks')
    write_doc('wrappers')



if __name__ == '__main__':
    main()
