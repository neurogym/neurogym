"""Programmatically generate jupyte notebooks."""

from pathlib import Path
import inspect
import importlib

import nbformat as nbf

import neurogym as ngym


def get_linenumber(m):
    """Get line number of a member."""
    try:
        return inspect.findsource(m[1])[1]
    except AttributeError:
        return -1


def get_members(modulename):
    """"Get all members defined in a module.

    Args:
        modulename: str, modulename

    Returns:
        members: list of (name, object) pairs
    """
    # Get functions from module
    # modulename = 'train_and_analysis_template'
    module = importlib.import_module(modulename)
    members = inspect.getmembers(module)  # get all members
    # members = [m for m in members if inspect.isfunction(m[1])]  # keep all functions
    m_tmp = list()
    for m in members:
        try:
            tmp_module = m[1].__module__
            if tmp_module == modulename:
                m_tmp.append(m)
        except AttributeError:
            pass

    members = m_tmp
    members.sort(key=get_linenumber)  # sort by line number
    return members


def func_to_script(code):
    # Remove indentation
    code = code.replace('\n    ', '\n')
    # Remove first line
    ind = code.find('\n')
    assert code[ind-1] == ':'  # end of def should be this
    code = code[ind+1:]
    
    # Search if there is a return
    ind = code.find('return')
    # Check this is the only return
    assert code.find('return', ind + 1) == -1
    
    # Remove the return line
    code = code[:ind]
    
    return code


COLABURL = 'https://colab.research.google.com/github/'
ROOTURL = 'neurogym/ngym_usage/blob/master/training/auto_notebooks/'


def auto_generate_notebook(envid, learning='supervised'):
    installation_code = "# ! pip install gym   # Install gym\n" +\
           "# ! git clone https://github.com/gyyang/neurogym.git  # Install " \
           "neurogym\n"\
           "# %cd neurogym/\n"\
           "# ! pip install -e .\n"

    if learning == 'supervised':
        modulename = 'supervised_train'
    elif learning == 'rl':
        modulename = 'RL_train'
        installation_code = '# %tensorflow_version 1.x\n' +\
                            '# ! pip install --upgrade stable-baselines  # ' \
                            'install latest version\n' + \
                            installation_code
    else:
        raise ValueError('Unknown learning', learning)
    installation_code = "# Uncomment following lines to install\n" + installation_code
    codeurl = ROOTURL + learning + '/'
    # From training code
    train_members = get_members(modulename)
    # Common analysis code
    analysis_members = get_members('train_and_analysis_template')
    members = train_members + analysis_members

    nb = nbf.v4.new_notebook()
    # Initial code block
    with open(modulename + '.py', 'r') as f:
        codelines = f.readlines()

    # Beginning text
    badge = 'https://colab.research.google.com/assets/colab-badge.svg'
    link = '{:s}{:s}{:s}.ipynb'.format(COLABURL, codeurl, envid)
    text = """[![Open In Colab]({:s})]({:s})""".format(badge, link)

    cells = [nbf.v4.new_markdown_cell(text)]

    # Local install if on colab
    text = '### Install packages if on Colab'
    cells += [nbf.v4.new_markdown_cell(text)]
    cells += [nbf.v4.new_code_cell(installation_code)]

    # Everything before first function/class
    text = '### Import packages'
    cells += [nbf.v4.new_markdown_cell(text)]

    code = ''.join(codelines[:get_linenumber(members[0])])
    code = code + "envid = '{:s}'".format(envid)
    cells += [nbf.v4.new_code_cell(code)]

    first_analysis = True
    func_to_script_list = ['train_network', 'run_network']
    for name, obj in members:
        code = inspect.getsource(obj)

        if name == 'train_network':  # TODO: Need to change to include RL
            text = '### Train network'
            cells += [nbf.v4.new_markdown_cell(text)]
        elif name == 'run_network':
            text = '### Run network after training for analysis'
            cells += [nbf.v4.new_markdown_cell(text)]
        elif name in ['Net', 'Model']:
            text = '### Define network'
            cells += [nbf.v4.new_markdown_cell(text)]

        if name in func_to_script_list:
            # Turn function into script
            code = func_to_script(code)

        if name.find('analysis_') == 0:  # starts with this
            if first_analysis:
                text = '### General analysis'
                cells += [nbf.v4.new_markdown_cell(text)]
                first_analysis = False
            # Add a line that's running this function
            code = code + '\n' + code[4:code.find('\n') - 1] # 4 for "def "

        cells.append(nbf.v4.new_code_cell(code))

    nb['cells'] = cells

    #     nb['cells'] = [nbf.v4.new_markdown_cell(text),
    #                nbf.v4.new_code_cell(code) ]

    fname = Path('.') / 'auto_notebooks' / learning / (envid + '.ipynb')
    nbf.write(nb, fname)


if __name__ == '__main__':
    all_envs = ngym.all_envs(tag='supervised')
    for envid in all_envs:
        auto_generate_notebook(envid, learning='supervised')

    all_envs = ngym.all_envs()
    for envid in all_envs:
        auto_generate_notebook(envid, learning='rl')