# NeuroGym

NeuroGym is comprehensive toolkit that allows training any network model on many established neuroscience tasks using Reinforcement Learning techniques. NeuroGym includes working memory tasks, value-based decision tasks and context-dependent perceptual categorization tasks. 

NeuroGym Inherites all functionalities from the machine learning toolkit Gym (OpenAI) and thus allows a wide range of well established machine learning algorithms to be easily trained on behavioral paradigms relevant for the neuroscience community. NeuroGym also incorporates several properties and functions (e.g. realistic time step or separation of training into trials) that are specific to the protocols used in neuroscience.
The toolkit also includes various modifier functions that greatly expand the space of available tasks. For instance, users can introduce trial-to-trial correlations onto any task. Also, tasks can be combined so as to test the capacity of a given model to perform two tasks simultaneously ([Molano-Mazon et al. CNS2019](https://www.cnsorg.org/cns-2019)). 

![alt tag](figures/pipeline.png)

### Prerequisites

* Python 3.5+
* Numpy.
* SciPy.
* Matplotlib.
* [Gym](https://gym.openai.com/).


### Example

The following example will run the Random Dots Motion task for 200 steps with random actions.

```
python /path-to-neurogym-toolbox/neurogym/ops/test_new.py --task1=RDM-v0  --n_steps=200 
```

Below we show a more elaborate example that requires installing [the openAi toolbox Baselines](https://github.com/openai/baselines) (a set of implementations of reinforcement learning algorithms). The example below uses the A2C algorithm to learn Random Dots Motion task:

```
python /path-to-baselines-toolbox/baselines/run.py --alg=a2c --env=RDM-v0 --network=lstm --nsteps=20 --nlstm=64 --stimEv=0.5 --pass_reward=True --pass_action=True --timing 200 200 300 400 200
```

**alg**: algorithm; **env**: task; **network**: type of network; **nsteps**: batch size; **nlstm**: number of units; **stimEv**: controls the difficulty of the experiment; **pass_reward**: if true, the task passes the previous reward as a part of the observation;  **pass_action**: if true, the task passes the previous action as a part of the observation; **timing**: duration of the different periods in a trial.


### Authors
* [Manuel Molano](https://github.com/manuelmolano).
* [Guangyu Robert Yang](https://github.com/gyyang).


