"""Programmatically generate jupyter notebooks."""  # noqa: INP001

import importlib
import inspect
from pathlib import Path
from typing import Any, Literal

import nbformat as nbf

import neurogym as ngym


def get_linenumber(m: tuple[str, Any]) -> int:
    """Get line number of a member."""
    try:
        return inspect.findsource(m[1])[1]
    except AttributeError:
        return -1


def get_members(modulename: str) -> list[tuple[str, Any]]:
    """Get all members defined in a module.

    Args:
        modulename: str, modulename

    Returns:
        members: list of (name, object) pairs
    """
    # Get functions from module
    module = importlib.import_module(modulename)
    members = inspect.getmembers(module)  # get all members
    # members = [m for m in members if inspect.isfunction(m[1])]  # keep all functions # noqa: ERA001
    m_tmp = []
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


def func_to_script(code: str) -> str:
    # Remove indentation
    code = code.replace("\n    ", "\n")
    # Remove first line
    ind = code.find("\n")
    if code[ind - 1] != ":":
        msg = "function definition line should end with a semi-colon"
        raise ValueError(msg)
    code = code[ind + 1 :]

    # Search if there is a return
    ind = code.find("return")
    # Check this is the only return
    if code.find("return", ind + 1) != -1:
        # FIXME: why this check? a function can have multiple (conditional) returns
        msg = "Multiple return statements found for function"
        raise ValueError(msg)

    return code[:ind]  # Remove the return line


def auto_generate_notebook(envid: str, learning: Literal["supervised", "reinforcement"] = "supervised") -> None:
    # Google Colab badge and link
    colab_url = "https://colab.research.google.com"
    destination_url = "github/neurogym/neurogym/docs/tutorials/auto_generated"
    link = f"{colab_url}/{destination_url}/{learning}/{envid}.ipynb"
    badge = f"{colab_url}/assets/colab-badge.svg"
    text = f"[![Open In Colab]({badge})]({link})"
    cells = [nbf.v4.new_markdown_cell(text)]

    # TODO: include option to add a custom introduction text
    text = f"##{envid}"

    # Installation instructions
    text = (
        "### Installation\n\n"
        "**Google Colab:** Uncomment and execute cell below when running this notebook on google colab.\n\n"
        "**Local:** Follow [these instructions](https://github.com/neurogym/neurogym?tab=readme-ov-file#installation)\n"
        "when running this notebook locally.\n"
    )
    cells += [nbf.v4.new_markdown_cell(text)]

    # Installation code
    installation_code = "# ! pip install neurogym"
    if learning == "supervised":
        modulename = "supervised_train"
    elif learning == "reinforcement":
        modulename = "RL_train"
        installation_code += "[rl]"
    else:
        msg = f"Unknown learning type: {learning}"
        raise ValueError(msg)
    cells += [nbf.v4.new_code_cell(installation_code)]

    # From training code
    train_members = get_members(modulename)
    # Common analysis code
    analysis_members = get_members("train_and_analysis_template")
    members = train_members + analysis_members

    nb = nbf.v4.new_notebook()
    # Initial code block
    with Path(f"{modulename}.py").open() as f:
        codelines = f.readlines()

    # Everything before first function/class
    text = "### Import packages"
    cells += [nbf.v4.new_markdown_cell(text)]

    code = "".join(codelines[: get_linenumber(members[0])])
    code = code + f"envid = '{envid}'"
    cells += [nbf.v4.new_code_cell(code)]

    first_analysis = True
    func_to_script_list = ["train_network", "run_network"]
    for name, obj in members:
        code = inspect.getsource(obj)

        if name == "train_network":  # TODO: Need to change to include RL
            text = "### Train network"
            cells += [nbf.v4.new_markdown_cell(text)]
        elif name == "run_network":
            text = "### Run network after training for analysis"
            cells += [nbf.v4.new_markdown_cell(text)]
        elif name in ["Net", "Model"]:
            text = "### Define network"
            cells += [nbf.v4.new_markdown_cell(text)]

        if name in func_to_script_list:
            # Turn function into script
            code = func_to_script(code)

        if name.find("analysis_") == 0:  # starts with this
            if first_analysis:
                text = "### General analysis"
                cells += [nbf.v4.new_markdown_cell(text)]
                first_analysis = False
            # Add a line that's running this function
            code = code + "\n" + code[4 : code.find("\n") - 1]  # 4 for "def "

        cells.append(nbf.v4.new_code_cell(code))

    nb["cells"] = cells

    #     nb['cells'] = [nbf.v4.new_markdown_cell(text),
    #                nbf.v4.new_code_cell(code) ]

    fname = Path("auto_notebooks") / learning / (envid + ".ipynb")
    nbf.write(nb, fname)


if __name__ == "__main__":
    all_envs = ngym.all_envs(tag="supervised")
    for envid in all_envs:
        auto_generate_notebook(envid, learning="supervised")

    all_envs = ngym.all_envs()
    for envid in all_envs:
        auto_generate_notebook(envid, learning="reinforcement")
