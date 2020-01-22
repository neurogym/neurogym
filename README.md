# NeuroGym

#### In Development. Tasks are subject to changes right now

NeuroGym is a comprehensive toolkit that allows training any network model on many established neuroscience tasks using Reinforcement Learning techniques. NeuroGym includes working memory tasks, value-based decision tasks and context-dependent perceptual categorization tasks. 

NeuroGym Inherites all functionalities from the machine learning toolkit Gym (OpenAI) and thus allows a wide range of well established machine learning algorithms to be easily trained on behavioral paradigms relevant for the neuroscience community. NeuroGym also incorporates several properties and functions (e.g. realistic time step or separation of training into trials) that are specific to the protocols used in neuroscience.
The toolkit also includes various modifier functions that greatly expand the space of available tasks. For instance, users can introduce trial-to-trial correlations onto any task. Also, tasks can be combined so as to test the capacity of a given model to perform two tasks simultaneously ([Molano-Mazon et al. CNS2019](https://www.cnsorg.org/cns-2019)). 

![alt tag](docs/pipeline.png)

### Prerequisites

* Python 3.5+
* Numpy.
* SciPy.
* Matplotlib.
* [Gym](https://gym.openai.com/).

### Installation

You can perform a minimal install of ``neurogym`` with:

    git clone https://github.com/gyyang/neurogym.git
    cd neurogym
    pip install -e .

### Implemented tasks
Currently implemented tasks can be found [here](https://github.com/gyyang/neurogym/blob/master/docs/envs.md).

### Example

In the folder examples we provide two scripts that show how to use Neurogym:

In the *a2c_example.ipynb* file we provide an example that installs the necessary toolboxes and trains the A3C algorithm [Mnih et al. 2016](https://arxiv.org/abs/1602.01783) on the Random Dots Motion algorithm.

Further, NeuroGym can also be used together with the openAi toolbox [Stable Baselines](https://github.com/hill-a/stable-baselines) (a set of implementations of reinforcement learning algorithms). The example below uses the A2C algorithm to learn Random Dots Motion task.


### Contributing

Contributing new tasks is easy. You can see a template of how a task should look like in the *superclass* ngym, which is thought to contain all functions that are common to all tasks:
```
class ngym(gym.Env):
    def __init__(self, dt=100):
        """
	initializes relevant variables (e.g. different periods durations)
	"""

    def step(self, action):
        """
        receives an action and returns a new observation, a reward, a flag variable
        indicating whether the experiment has ended and a dictionary with
        useful information (info). Note that a great part of this, is done by 
	_step(action) (see below). step() aditionally calls the function _new_trial
	if the current trial is done info['new_trial']==True).
        """
        return obs, rew, done, info

    def reset(self):
        """
        resets the experiment
        """
        return obs

    def render(self, mode='human'):
        """
        plots relevant variables/parameters (so far not used)
        """
        pass

    # Auxiliary functions
    def seed(self, seed=None):
        self.rng = random
        self.rng.seed(seed)
        return [seed]

    def _step(self, action):
        """
        receives an action and returns a new state, a reward, a flag variable
        indicating whether the experiment has ended and a dictionary with
        useful information
        """
        return obs, rew, done, info

    def new_trial(self, **kwargs):
        """Starts a new trial within the current experiment.
        Returns trial_info: a dictionary of trial information
        """
        return trial_info

    def in_epoch(self, t, epoch):
        """Check if t is in epoch."""

```

All tasks call ngym as their superclass and so they inherite all above functions. However you will likely want to modify some of them in your task. If you want to see how this is done in an actual task, the Random Dots Motion task (RDM) is probably the one containing more comments.

### Wrappers

Wrappers are short scripts that allow introducing modifications the original tasks. For instance, the Random Dots Motion task can be transformed into a reaction time task by passing it through the *reaction_time* wrapper. Alternatively, the *combine* wrapper allows training an agent in two different tasks simultaneously. 



### Authors
* [Manuel Molano](https://github.com/manuelmolano).
* [Guangyu Robert Yang](https://github.com/gyyang).


