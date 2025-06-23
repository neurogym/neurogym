"""Formatting information about envs and wrappers."""

import inspect

from neurogym.core import METADATA_DEF_KEYS, env_string
from neurogym.envs.registration import ALL_ENVS, all_tags, make
from neurogym.utils.logging import logger
from neurogym.wrappers import ALL_WRAPPERS, all_wrappers


def show_all_tasks() -> None:
    logger.info("Available tasks:")
    for task in sorted(ALL_ENVS):
        logger.info(task)


def show_all_wrappers() -> None:
    logger.info("Available wrappers:")
    for wrapper in all_wrappers():
        logger.info(wrapper)


def info(env=None, show_code=False):
    """Script to get envs info."""
    string = ""
    env_name = env
    env = make(env)
    # remove extra wrappers (make can add a OrderEnforcer wrapper)
    env = env.unwrapped
    string = env_string(env)
    # show source code
    if show_code:
        string += """\n#### Source code #### \n\n"""
        env_ref = ALL_ENVS[env_name]
        from_, class_ = env_ref.split(":")
        imported = getattr(__import__(from_, fromlist=[class_]), class_)
        lines = inspect.getsource(imported)
        string += lines + "\n\n"
    return string


def info_wrapper(wrapper=None, show_code=False):
    """Script to get wrappers info."""
    string = ""

    wrapp_ref = ALL_WRAPPERS[wrapper]
    from_, class_ = wrapp_ref.split(":")
    imported = getattr(__import__(from_, fromlist=[class_]), class_)
    metadata = imported.metadata

    if not isinstance(metadata, dict):
        metadata = {}

    string += f"### {wrapper}\n\n"
    paper_name = metadata.get("paper_name", None)
    paper_link = metadata.get("paper_link", None)
    wrapper_description = metadata.get("description", None) or "Missing description"
    string += f"Logic: {wrapper_description}\n\n"
    if paper_name is not None:
        string += "Reference paper \n\n"
        if paper_link is None:
            string += f"{paper_name}\n\n"
        else:
            string += f"[{paper_name}]({paper_link})\n\n"
    # add extra info
    other_info = list(set(metadata.keys()) - set(METADATA_DEF_KEYS))
    if len(other_info) > 0:
        string += "Input parameters: \n\n"
        for key in other_info:
            string += f"{key} : {metadata[key]}\n\n"

    # show source code
    if show_code:
        string += """\n#### Source code #### \n\n"""
        lines = inspect.getsource(imported)
        string += lines + "\n\n"

    return string


def show_all_tags():
    logger.info("Available tags:")
    for tag in all_tags():
        logger.info(tag)
