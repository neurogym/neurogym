**Status:** In Development. All tasks are subject to changes right now.

# NeuroGym

|     Badges     |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| :------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  **fairness**  | [![OpenSSF Best Practices](https://www.bestpractices.dev/projects/9839/badge)](https://www.bestpractices.dev/projects/9839) [![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F-green)](https://fair-software.eu)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|  **package**   | [![PyPI version](https://badge.fury.io/py/neurogym.svg)](https://badge.fury.io/py/neurogym)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|    **docs**    | [![Documentation](https://img.shields.io/badge/docs-mkdocs-259482)](https://neurogym.github.io/neurogym/latest/) [![RSD](https://img.shields.io/badge/RSD-neurogym-3dffff)](https://research-software-directory.org/software/neurogym) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14508987.svg)](https://doi.org/10.5281/zenodo.14508987)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
|   **tests**    | [![build](https://github.com/neurogym/neurogym/actions/workflows/build.yml/badge.svg)](https://github.com/neurogym/neurogym/actions/workflows/build.yml) [![sonarcloud](https://github.com/neurogym/neurogym/actions/workflows/sonarcloud.yml/badge.svg)](https://github.com/neurogym/neurogym/actions/workflows/sonarcloud.yml) [![linkspector](https://github.com/neurogym/neurogym/actions/workflows/linkspector.yml/badge.svg)](https://github.com/neurogym/neurogym/actions/workflows/linkspector.yml) [![cffconvert](https://github.com/neurogym/neurogym/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/neurogym/neurogym/actions/workflows/cffconvert.yml) [![linting](https://github.com/neurogym/neurogym/actions/workflows/linting.yml/badge.svg)](https://github.com/neurogym/neurogym/actions/workflows/linting.yml) [![static-typing](https://github.com/neurogym/neurogym/actions/workflows/static-typing.yml/badge.svg)](https://github.com/neurogym/neurogym/actions/workflows/static-typing.yml) [![workflow scq badge](https://sonarcloud.io/api/project_badges/measure?project=neurogym_neurogym&metric=alert_status)](https://sonarcloud.io/dashboard?id=neurogym_neurogym) [![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=neurogym_neurogym&metric=coverage)](https://sonarcloud.io/dashboard?id=neurogym_neurogym) |
| **running on** | [![ubuntu](https://img.shields.io/badge/ubuntu-latest-8A2BE2?style=plastic)](https://github.com/actions/runner-images?tab=readme-ov-file#available-images)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|  **license**   | [![github license badge](https://img.shields.io/github/license/ANNUBS/annubes)](https://github.com/ANNUBS/annubes?tab=Apache-2.0-1-ov-file)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |

NeuroGym is a curated collection of neuroscience tasks with a common interface. The goal is to facilitate the training of neural network models on neuroscience tasks.

- [NeuroGym](#neurogym)
  - [Installation](#installation)
    - [Psychopy installation](#psychopy-installation)
  - [Tasks](#tasks)
  - [Wrappers](#wrappers)
  - [Examples](#examples)
  - [Custom tasks](#custom-tasks)
  - [Acknowledgements](#acknowledgements)

NeuroGym inherits from the machine learning toolkit [Gymnasium](https://gymnasium.farama.org/), a maintained fork of [OpenAI’s Gym library](https://github.com/openai/gym). It allows a wide range of well established machine learning algorithms to be easily trained on behavioral paradigms relevant for the neuroscience community.
NeuroGym also incorporates several properties and functions (e.g. continuous-time and trial-based tasks) that are important for neuroscience applications. The toolkit also includes various modifier functions that allow easy configuration of new tasks.

![alt tag](https://github.com/neurogym/neurogym/blob/main/docs/pipeline.png)

### Installation

Create and activate a virtual environment to install the current package, e.g. using
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) (please refer to their
site for questions about creating the environment):

```bash
conda activate # ensures you are in the base environment
conda create -n neurogym python=3.11
conda activate neurogym
```

Then install neurogym as follows:

```bash
git clone https://github.com/neurogym/neurogym.git
cd neurogym
pip install -e .
```

#### Psychopy installation

If you need psychopy for your project, additionally run

```bash
pip install psychopy
```

### Tasks

Currently implemented tasks can be found [here](https://neurogym.github.io/envs/index.html).

### Wrappers

Wrappers (see [list](https://github.com/gyyang/neurogym/blob/master/docs/wrappers.md))
are short scripts that allow introducing modifications the original tasks. For instance, the Random Dots Motion task can be transformed into a reaction time task by passing it through the _reaction_time_ wrapper. Alternatively, the _combine_ wrapper allows training an agent in two different tasks simultaneously.

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

Other contributors (listed in chronological order)

- [Marta Fradera](https://github.com/martafradera)
- [Jordi Pastor](https://github.com/pastorjordi)
- [Jeremy Forest](https://github.com/jeremyforest)
- [Ru-Yuan Zhang](https://github.com/ruyuanzhang)
