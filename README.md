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

NeuroGym can be used together with the openAi toolbox [Stable Baselines](https://github.com/hill-a/stable-baselines) (a set of implementations of reinforcement learning algorithms). [Here](https://github.com/gyyang/neurogym/blob/master/neurogym/examples/example_NeuroGym_stable_baselines.ipynb) you will find an example of how to do it.


### Contributing


### Wrappers

Wrappers are short scripts that allow introducing modifications the original tasks. For instance, the Random Dots Motion task can be transformed into a reaction time task by passing it through the *reaction_time* wrapper. Alternatively, the *combine* wrapper allows training an agent in two different tasks simultaneously. 



### Authors
* [Manuel Molano](https://github.com/manuelmolano).
* [Guangyu Robert Yang](https://github.com/gyyang).


