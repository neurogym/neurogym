# NeuroGym

|     Badges     |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| :------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  **fairness**  | [![OpenSSF Best Practices](https://www.bestpractices.dev/projects/9839/badge)](https://www.bestpractices.dev/projects/9839) [![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F-green)](https://fair-software.eu)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|  **package**   | [![PyPI version](https://badge.fury.io/py/neurogym.svg)](https://badge.fury.io/py/neurogym)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|    **docs**    | [![Documentation](https://img.shields.io/badge/docs-mkdocs-259482)](https://neurogym.github.io/neurogym/latest/) [![RSD](https://img.shields.io/badge/RSD-neurogym-3dffff)](https://research-software-directory.org/software/neurogym) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14508986.svg)](https://doi.org/10.5281/zenodo.14508986)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
|   **tests**    | [![build](https://github.com/neurogym/neurogym/actions/workflows/build.yml/badge.svg)](https://github.com/neurogym/neurogym/actions/workflows/build.yml) [![sonarcloud](https://github.com/neurogym/neurogym/actions/workflows/sonarcloud.yml/badge.svg)](https://github.com/neurogym/neurogym/actions/workflows/sonarcloud.yml) [![linkspector](https://github.com/neurogym/neurogym/actions/workflows/linkspector.yml/badge.svg)](https://github.com/neurogym/neurogym/actions/workflows/linkspector.yml) [![cffconvert](https://github.com/neurogym/neurogym/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/neurogym/neurogym/actions/workflows/cffconvert.yml) [![linting](https://github.com/neurogym/neurogym/actions/workflows/linting.yml/badge.svg)](https://github.com/neurogym/neurogym/actions/workflows/linting.yml) [![static-typing](https://github.com/neurogym/neurogym/actions/workflows/static-typing.yml/badge.svg)](https://github.com/neurogym/neurogym/actions/workflows/static-typing.yml) [![workflow scq badge](https://sonarcloud.io/api/project_badges/measure?project=neurogym_neurogym&metric=alert_status)](https://sonarcloud.io/dashboard?id=neurogym_neurogym) [![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=neurogym_neurogym&metric=coverage)](https://sonarcloud.io/dashboard?id=neurogym_neurogym) |
| **running on** | [![ubuntu](https://img.shields.io/badge/ubuntu-latest-8A2BE2?style=plastic)](https://github.com/actions/runner-images?tab=readme-ov-file#available-images)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|  **license**   | [![github license badge](https://img.shields.io/github/license/neurogym/neurogym)](https://github.com/neurogym/neurogym?tab=Apache-2.0-1-ov-file)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |

NeuroGym is a curated collection of neuroscience tasks with a common interface. The goal is to facilitate the training of neural network models on neuroscience tasks.

- [NeuroGym](#neurogym)
  - [Installation](#installation)
    - [Step 1: Create a virtual environment](#step-1-create-a-virtual-environment)
    - [Step 2: Install NeuroGym](#step-2-install-neurogym)
      - [Step 2b: Install in Editable/Development Mode](#step-2b-install-in-editabledevelopment-mode)
    - [Step 3 (Optional): Psychopy installation](#step-3-optional-psychopy-installation)
  - [Tasks](#tasks)
  - [Wrappers](#wrappers)
  - [Configuration](#configuration)
    - [1. From a TOML file](#1-from-a-toml-file)
    - [2. With Python class](#2-with-python-class)
    - [3. With a dictionary](#3-with-a-dictionary)
  - [Examples](#examples)
  - [Custom tasks](#custom-tasks)
  - [Acknowledgements](#acknowledgements)

NeuroGym inherits from the machine learning toolkit [Gymnasium](https://gymnasium.farama.org/), a maintained fork of [OpenAI’s Gym library](https://github.com/openai/gym). It allows a wide range of well established machine learning algorithms to be easily trained on behavioral paradigms relevant for the neuroscience community.
NeuroGym also incorporates several properties and functions (e.g. continuous-time and trial-based tasks) that are important for neuroscience applications. The toolkit also includes various modifier functions that allow easy configuration of new tasks.

Please see our extended project [documentation](https://neurogym.github.io/neurogym/latest/) for additional details.

![alt tag](https://github.com/neurogym/neurogym/blob/main/docs/pipeline.png)

### Installation

#### Step 1: Create a virtual environment

Create and activate a virtual environment to install the current package, e.g. using
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) (please refer to their
site for questions about creating the environment):

```bash
conda activate # ensures you are in the base environment
conda create -n neurogym python=3.11 -y
conda activate neurogym
```

#### Step 2: Install NeuroGym

You can install the latest stable release of `neurogym` using pip:

```bash
pip install neurogym
```

If you plan to use reinforcement learning (RL) features based on Stable-Baselines3, install the RL extra dependencies:

```bash
pip install neurogym[rl]
```

##### Step 2b: Install in Editable/Development Mode

If you want to contribute to NeuroGym or always work with the latest updates from the source code, install it in editable mode:

```bash
git clone https://github.com/neurogym/neurogym.git
cd neurogym
pip install -e .
```

This links your local code changes directly to your Python environment without needing to reinstall after every edit.

If you also want RL and development tools (for testing, linting, and documentation), install with:

```bash
pip install -e .[rl,dev]
```

#### Step 3 (Optional): Psychopy installation

**NOTE**: psycohopy installation is currently not working

If you need psychopy for your project, additionally run

```bash
pip install psychopy
```

### Tasks

Currently implemented tasks can be found [here](https://neurogym.github.io/envs/index.html).

### Wrappers

Wrappers (see [their docs](https://neurogym.github.io/neurogym/latest/api/wrappers/))
are short scripts that allow introducing modifications the original tasks. For instance, the Random Dots Motion task can be transformed into a reaction time task by passing it through the _reaction_time_ wrapper. Alternatively, the _combine_ wrapper allows training an agent in two different tasks simultaneously.

### Configuration

🧪 **Beta Feature** — The configuration system is **optional** and currently **under development**. You can still instantiate environments, agents, and wrappers with direct parameters.
It is only used in a small portion of the codebase and is not required for typical usage.
See the [`demo.ipynb`](docs/examples/demo.ipynb) notebook for the only current example of this system in action.

NeuroGym includes a flexible configuration mechanism using [`Pydantic Settings`](https://docs.pydantic.dev/latest/concepts/pydantic_settings/), allowing configuration via TOML files, Python objects, or plain dictionaries.

Using a TOML file can be especially useful for sharing experiment configurations in a portable way (e.g., sending `config.toml` to a colleague), reliably saving and loading experiment setups, and easily switching between multiple configurations for the same environment by changing just one line of code. While the system isn't at that stage yet, these are intended future capabilities.

#### 1. From a TOML file

Create a `config.toml` file (see [template](docs/examples/config.toml)) and load it:

```python
from neurogym import Config
config = Config('path/to/config.toml')
```

You can then pass this config to any component that supports it:

```python
from neurogym.wrappers import monitor
env = gym.make('GoNogo-v0')
env = monitor.Monitor(env, config=config)
```

Or directly pass the path:

```python
env = monitor.Monitor(env, config='path/to/config.toml')
```

#### 2. With Python class

```python
from neurogym import Config
config = Config(
    local_dir="logs/",
    env={"name": "GoNogo-v0"},
    monitor={"name": "MyMonitor"}
)
```

#### 3. With a dictionary

```python
from neurogym import Config
config_dict = {
    "env": {"name": "GoNogo-v0"},
    "monitor": {
        "name": "MyMonitor",
        "plot": {"trigger": "step", "value": 500, "create": True}
    },
    "local_dir": "./outputs"
}
config = Config.model_validate(config_dict)
```

### Examples

NeuroGym is compatible with most packages that use gymnasium.
In this [example](https://github.com/neurogym/neurogym/blob/main/docs/examples/example_neurogym_rl.ipynb) jupyter notebook we show how to train a neural network with RL algorithms using the [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) toolbox.

### Custom tasks

Creating custom new tasks should be easy. You can contribute tasks using the regular gymnasium format. If your task has a trial/period structure, this [template](https://github.com/neurogym/neurogym/blob/main/docs/examples/template.py) provides the basic structure that we recommend a task to have:

```python
from gymnasium import spaces
import neurogym as ngym

class YourTask(ngym.PeriodEnv):
    metadata = {}

    def __init__(self, dt=100, timing=None, extra_input_param=None):
        super().__init__(dt=dt)


    def new_trial(self, **kwargs):
        """
        new_trial() is called when a trial ends to generate the next trial.
        Here you have to set:
        The trial periods: fixation, stimulus...
        Optionally, you can set:
        The ground truth: the correct answer for the created trial.
        """

    def _step(self, action):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has terminated, terminated
                See more at https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/#termination
            a boolean variable indicating whether the experiment has been truncated, truncated
                See more at https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/#truncation
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """

        return obs, reward, terminated, truncated, {'new_trial': new_trial, 'gt': gt}

```

### Acknowledgements

For the authors of the package, please refer to the zenodo DOI at the top of the page.
