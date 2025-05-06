# NeuroGym

## Tasks

Currently implemented tasks can be found [here](https://neurogym.github.io/neurogym/latest/api/envs/).

## Wrappers

Wrappers (see [here](https://neurogym.github.io/neurogym/latest/api/wrappers/))
are short scripts that allow introducing modifications the original tasks. For instance, the Random Dots Motion task can be transformed into a reaction time task by passing it through the _reaction_time_ wrapper. Alternatively, the _combine_ wrapper allows training an agent in two different tasks simultaneously.

## Configuration

🧪 **Beta Feature** — The configuration system is **optional** and currently **under development**. You can still instantiate environments, agents, and wrappers with direct parameters.
It is only used in a small portion of the codebase and is not required for typical usage.
See the [`demo.ipynb`](examples/demo.ipynb) notebook for the only current example of this system in action.

NeuroGym includes a flexible configuration mechanism using [`Pydantic Settings`](https://docs.pydantic.dev/latest/concepts/pydantic_settings/), allowing configuration via TOML files, Python objects, or plain dictionaries.

Using a TOML file can be especially useful for sharing experiment configurations in a portable way (e.g., sending `config.toml` to a colleague), reliably saving and loading experiment setups, and easily switching between multiple configurations for the same environment by changing just one line of code. While the system isn't at that stage yet, these are intended future capabilities.

### 1. From a TOML file

Create a `config.toml` file (see [template](examples/config.toml)) and load it:

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

### 2. With Python class

```python
from neurogym import Config
config = Config(
    local_dir="logs/",
    env={"name": "GoNogo-v0"},
    monitor={"name": "MyMonitor"}
)
```

### 3. With a dictionary

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

## Examples

NeuroGym is compatible with most packages that use gymnasium.
In this [example](https://github.com/gyyang/neurogym/blob/master/examples/example_neurogym_rl.ipynb) jupyter notebook we show how to train a neural network with reinforcement learning algorithms using the [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) toolbox.

## Custom tasks

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
