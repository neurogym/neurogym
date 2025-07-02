"""Formatting information about envs and wrappers."""

import gymnasium as gym

from neurogym.core import METADATA_DEF_KEYS, env_string
from neurogym.envs.registration import ALL_ENVS, all_envs, all_tags, make
from neurogym.utils.logging import logger
from neurogym.wrappers import ALL_WRAPPERS, all_wrappers


def show_all_tasks(tag: str | None = None) -> None:
    """Show all available tasks in neurogym.

    Args:
        tag: If provided, only show tasks with this tag.
    """
    if not tag:
        logger.info("Available tasks:", color="green")
    else:
        logger.info(f"Available tasks with tag '{tag}':", color="green")

    for task in all_envs(tag=tag):
        logger.info(task)


def show_all_wrappers() -> None:
    """Show all available wrappers in neurogym."""
    logger.info("Available wrappers:", color="green")
    for wrapper in all_wrappers():
        logger.info(wrapper)


def show_all_tags():
    """Show all available tags in neurogym."""
    logger.info("Available tags:", color="green")
    for tag in all_tags():
        logger.info(tag)


def show_info(obj_: str | gym.Env) -> None:
    """Show information about an environment or a wrapper.

    Using the built-in logger.

    Args:
        obj_: the environment or wrapper to show information about.
    """
    if isinstance(obj_, str):
        if obj_ in ALL_ENVS:
            _env_info(env=make(obj_))
        elif obj_ in ALL_WRAPPERS:
            _wrap_info(obj_)
        else:
            msg = f"Unknown environment or wrapper: {obj_}"
            raise ValueError(msg)

    elif isinstance(obj_, gym.Env):
        _env_info(obj_)

    else:
        msg = f"Expected a str or gym.Env, got {type(obj_)}"
        raise TypeError(msg)


def _env_info(env: gym.Env) -> None:
    """Show information about an environment."""
    env = env.unwrapped  # remove extra wrappers (make can add a OrderEnforcer wrapper)
    logger.info("Info for environment:" + env_string(env)[3:])


def _wrap_info(wrapper: str) -> None:
    """Show information about a wrapper."""
    logger.info(f"Info for wrapper: {wrapper}")

    wrapp_ref = ALL_WRAPPERS[wrapper]
    from_, class_ = wrapp_ref.split(":")
    imported = getattr(__import__(from_, fromlist=[class_]), class_)
    metadata = imported.metadata
    if not isinstance(metadata, dict):
        metadata = {}

    wrapper_description = metadata.get("description", None) or "Missing description"
    logger.info(f"Logic: {wrapper_description}")

    paper_name = metadata.get("paper_name", None)
    paper_link = metadata.get("paper_link", None)
    if paper_name is not None:
        reference = "Reference paper: "
        if paper_link is None:
            reference += f"{paper_name}"
        else:
            reference += f"[{paper_name}]({paper_link})"
        logger.info(reference)

    other_info = list(set(metadata.keys()) - set(METADATA_DEF_KEYS))
    if len(other_info) > 0:
        logger.info("Input parameters:")
        for key in other_info:
            logger.info(f"{key} : {metadata[key]}")
