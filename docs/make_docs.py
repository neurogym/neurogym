"""Script to make environment md"""

from collections import defaultdict

import gym
import neurogym as ngym
from neurogym.meta.info import info, info_wrapper


SOURCE_ROOT = 'https://github.com/gyyang/neurogym/blob/master/'


def write_doc(write_type):
    if write_type == 'tasks':
        all_items = ngym.all_tasks()
        info_fn = info
        fname = 'envs.md'
    elif write_type == 'wrappers':
        all_items = ngym.all_wrappers()
        info_fn = info_wrapper
        fname = 'wrappers.md'
    else:
        raise ValueError

    string = ''
    names = ''
    counter = 0
    link_dict = dict()
    for name in sorted(all_items.keys()):
        try:
            string += '___\n\n'
            string += info_fn(name)
            print(string)

            # Using github's automatic link to section titles
            if write_type == 'tasks':
                env = gym.make(name)
                link = type(env).__name__
            else:
                link = name
            link = link.lower().replace(' ', '-')
            link_dict[name] = link

            names += '[{:s}](#{:s})\n\n'.format(name, link)
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
    if write_type == 'tasks':
        all_tags = []
        tag_dict = defaultdict(list)
        for name in ngym.all_tasks():
            env = gym.make(name)
            tag_list = env.metadata.get('tags', [])
            all_tags += tag_list
            for tag in tag_list:
                tag_dict[tag].append(name)
        all_tags = set(all_tags)
        string_tag = '___\n\n### Tags ### \n\n'
        for tag in sorted(all_tags):
            string_tag += '### {:s} \n\n'.format(tag)
            for name in tag_dict[tag]:
                string_tag += '[{:s}]({:s})'.format(name, link_dict[name])
                string_tag += '\n\n'

        string = string + string_tag

    with open(fname, 'w') as f:
        f.write('* Under development, details subject to change\n\n')
        f.write(string)


def main():
    write_doc('tasks')
    write_doc('wrappers')



if __name__ == '__main__':
    main()
