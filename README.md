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
    - [Psychopy installation](#psychopy-installation)
  - [Tasks](#tasks)
  - [Wrappers](#wrappers)
  - [Configuration](#configuration)
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

#### Step 2: Install neurogym

Then install the latest version of `neurogym` as follows:

```bash
pip install neurogym
```

##### Step 2b: Install editable package

Alternatively, get the latest updates by cloning the repo and installing the editable version of neurogym, by replacing
step 2 above by:

```bash
git clone https://github.com/neurogym/neurogym.git
cd neurogym
pip install -e .
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

Wrappers (see [list](https://github.com/gyyang/neurogym/blob/master/docs/wrappers.md))
are short scripts that allow introducing modifications the original tasks. For instance, the Random Dots Motion task can be transformed into a reaction time task by passing it through the _reaction_time_ wrapper. Alternatively, the _combine_ wrapper allows training an agent in two different tasks simultaneously.

### Configuration
`Neurogym` employs a configuration system based on [`Pydantic Settings`](https://docs.pydantic.dev/latest/concepts/pydantic_settings/). You can use a TOML configuration file to create a configuration that can be used throughout the library, for instance, in existing and custom environments (cf. [`neurogym/conf/conf.py`](neurogym/conf/conf.py)). The supplied configuration template ([`docs/examples/conf.toml`](docs/examples/conf.toml)) contains all the currently available options, which mirror the default options in the code.

All options in the configuration code have default values. Rather than changing the values in the code, you can use a TOML file to override the defaults. To do that, just copy the [template configuration](docs/examples/conf.toml), change the values of any options that you need to customise, and save it somewhere convenient. You can pass that file to the `Conf` class to create a custom configuration vobject:

```python
from neurogym import Conf
conf = Conf('<path-to-TOML-file>')
```

The resulting object can be used on its own, or it can be passed to a wrapper (such as a monitor). It is also possible to pass the path to the TOML configuration file directly to the monitor when instantiating it:

```python
from neurogym.wrappers import monitor
env = gym.make('GoNogo-v0')
env = monitor.Monitor(env, conf="conf.toml") # You can also pass a conf object instead of a path.
```

Please refer to the [template configuration](docs/examples/conf.toml) for more details about the available options. The goal is to implement as many configurable options as possible in this configuration system in order to allow for environments, agents, experiments and so on to be instantiated just by passing a configuration file.

### Examples

NeuroGym is compatible with most packages that use gymnasium.
In this [example](https://github.com/gyyang/neurogym/blob/master/examples/example_neurogym_rl.ipynb) jupyter notebook we show how to train a neural network with reinforcement learning algorithms using the [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) toolbox.

### Custom tasks

Creating custom new tasks should be easy. You can contribute tasks using the regular gymnasium format. If your task has a trial/period structure, this [template](https://github.com/gyyang/neurogym/blob/master/examples/template.py) provides the basic structure that we recommend a task to have:

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
