**Status:** In Development. All tasks are subject to changes right now.

# NeuroGym

NeuroGym is a curated collection of neuroscience tasks with a common interface.
The goal is to facilitate training of neural network models on neuroscience tasks. 

Documentation: https://neurogym.github.io/
- [NeuroGym](#neurogym)
        - [Installation](#installation)
        - [Tasks](#tasks)
        - [Wrappers](#wrappers)
        - [Examples](#examples)
        - [Contributing](#contributing)
        - [Authors](#authors)

NeuroGym inherits from the machine learning toolkit [Gymnasium](https://gymnasium.farama.org/), a maintained fork of [OpenAI’s Gym library](https://github.com/openai/gym). It allows a wide range of well established machine learning algorithms to be easily trained on behavioral paradigms relevant for the neuroscience community. 
NeuroGym also incorporates several properties and functions (e.g. continuous-time and trial-based tasks) that are important for neuroscience applications.
The toolkit also includes various modifier functions that allow easy configuration of new tasks. 

![alt tag](docs/pipeline.png)

### Installation

You can perform a minimal install of ``neurogym`` with:

    git clone https://github.com/neurogym/neurogym.git
    cd neurogym
    pip install -e .
    
Or a full install by replacing the last command with ``pip install -e '.[all]'``

### Tasks
Currently implemented tasks can be found [here](https://github.com/gyyang/neurogym/blob/master/docs/envs.md).

### Wrappers
Wrappers (see [list](https://github.com/gyyang/neurogym/blob/master/docs/wrappers.md))
are short scripts that allow introducing modifications the original tasks. For instance, the Random Dots Motion task can be transformed into a reaction time task by passing it through the *reaction_time* wrapper. Alternatively, the *combine* wrapper allows training an agent in two different tasks simultaneously. 

### Examples

NeuroGym is compatible with most packages that use gymnasium. 
In this [example](https://github.com/gyyang/neurogym/blob/master/examples/example_neurogym_rl.ipynb) jupyter notebook we show how to train a neural network with reinforcement learning algorithms using the [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) toolbox.


### Contributing
Contributing new tasks should be easy. You can contribute tasks using the regular gymnasium format. If your task has a trial/period structure,
this [template](https://github.com/gyyang/neurogym/blob/master/examples/template.py) provides the basic structure that we recommend a task to have:

```
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




### Authors
* Contact

    [Manuel Molano](https://github.com/manuelmolano) (manuelmolanomazon@gmail.com).
    [Guangyu Robert Yang](https://github.com/gyyang) (gyyang.neuro@gmail.com).

* Contributors (listed in chronological order)

    [Marta Fradera](https://github.com/martafradera),
    [Jordi Pastor](https://github.com/pastorjordi),
    [Jeremy Forest](https://github.com/jeremyforest),
    [Ru-Yuan Zhang](https://github.com/ruyuanzhang)
