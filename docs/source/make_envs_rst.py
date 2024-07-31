from pathlib import Path

import gymnasium as gym
import httplib2
import matplotlib.pyplot as plt
import numpy as np

import neurogym as ngym
from neurogym.envs.registration import ALL_ENVS
from neurogym.wrappers import ALL_WRAPPERS

ENV_IGNORE = ["Null-v0"]
all_envs = {}
for key, val in sorted(ALL_ENVS.items()):
    if key in ENV_IGNORE:
        continue
    all_envs[key] = val


def make_env_images() -> None:
    envs = all_envs.keys()
    for env_name in envs:
        print("Make image for env", env_name)
        env = ngym.make(env_name, dt=20)
        action = np.zeros_like(env.action_space.sample())
        fname = Path(__file__).parent / "_static" / (env_name + "_examplerun")
        ngym.utils.plot_env(env, num_trials=2, def_act=action, fname=fname)
        plt.close()


SUPERVISEDURL = "neurogym/ngym_usage/blob/master/training/auto_notebooks/supervised/"
RLURL = "neurogym/ngym_usage/blob/master/training/auto_notebooks/rl/"
COLABURL = "https://colab.research.google.com/github/"


def _url_exist(url):
    """Check if this url exists."""
    h = httplib2.Http()
    resp = h.request(url, "HEAD")
    return int(resp[0]["status"]) < 400


def make_envs() -> None:
    # Make envs/index.rst
    string = "Environments\n"
    string += "===================================\n\n"
    string += ".. toctree::\n"
    string += "    :maxdepth: 1\n\n"
    for key in all_envs:
        string += " " * 4 + f"{key}\n"
    with open(Path(__file__).parent / "envs" / "index.rst", "w") as f:
        f.write(string)

    for key, val in all_envs.items():
        string = ""
        string += key + "\n" + "-" * 50 + "\n"
        string += ".. autoclass:: " + val.split(":")[0] + "." + val.split(":")[1] + "\n"
        string += "    :members:\n"
        string += "    :exclude-members: new_trial\n\n"

        env = gym.make(key)
        # Add paper
        paper_name = env.metadata.get("paper_name", "")
        paper_link = env.metadata.get("paper_link", "")
        if paper_name:
            string += "    Reference paper\n"
            paper_name = paper_name.replace("\n", " ")
            string += f"        `{paper_name} <{paper_link}>`__\n\n"

        # Add tags
        string += "    Tags\n"
        for tag in env.metadata.get("tags", []):
            string += f"        :ref:`tag-{tag}`, "
        string = string[:-2]
        string += "\n\n"

        # Add optional link to training and analysis code
        names = ["Supervised learning", "Reinforcement learning"]
        for baseurl, name in zip([SUPERVISEDURL, RLURL], names):
            url = f"https://github.com/{baseurl}{key}.ipynb"
            if _url_exist(url):
                string += " " * 4 + f"{name} and analysis of this task\n"
                link = f"{COLABURL}{baseurl}{key}.ipynb"
                string += " " * 8 + f"`[Open in colab] <{link}>`_\n"
                string += " " * 8 + f"`[Jupyter notebook Source] <{url}>`_\n"

        # Add image
        string += "    Sample run\n"
        image_path = Path("_static") / (key + "_examplerun.tmp")

        suffix = None
        _image_path = Path(__file__).parent / image_path
        for s in [".png", ".mp4"]:  # Check suffix
            if _image_path.with_suffix(s).exists():
                suffix = s
                break

        if suffix is not None:
            image_path = image_path.with_suffix(suffix).as_posix()
            if suffix == ".png":
                string += " " * 8 + f".. image:: ../{image_path}\n"
                string += " " * 12 + ":width: 600\n\n"
            elif suffix == ".mp4":
                string += " " * 8 + f".. video:: ../{image_path}\n"
                string += " " * 12 + ":width: 300\n"
                string += " " * 12 + ":height: 300\n"
                string += " " * 12 + ":loop:\n"

        with open(Path(__file__).parent / "envs" / (key + ".rst"), "w") as f:
            f.write(string)


def make_tags() -> None:
    string = "Tags\n"
    string += "===================================\n\n"

    all_tags = ngym.all_tags()

    for tag in sorted(all_tags):
        string += f".. _tag-{tag}:\n\n"
        string += tag + "\n--------------------------------\n"
        for env in ngym.all_envs(tag=tag):
            if env in ENV_IGNORE:
                continue
            string += "    :class:`{:s} <{:s}>`\n".format(
                env,
                all_envs[env].replace(":", "."),
            )
        string += "\n"
    with open(Path(__file__).parent / "tags.rst", "w") as f:
        f.write(string)

    string = "Wrappers\n"
    string += "===================================\n\n"

    for key, val in ALL_WRAPPERS.items():
        string += key + "\n" + "-" * 50 + "\n"
        string += ".. autoclass:: " + val.split(":")[0] + "." + val.split(":")[1] + "\n"
        string += "    :members:\n"
        string += "    :exclude-members: new_trial\n\n"

    with open(Path(__file__).parent / "wrappers.rst", "w") as f:
        f.write(string)


def main() -> None:
    make_env_images()
    make_envs()
    make_tags()


if __name__ == "__main__":
    main()
