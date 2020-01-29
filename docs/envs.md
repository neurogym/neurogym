* Under development, details subject to change

### List of environments implemented

* 24 tasks implemented so far.

### AngleReproduction task ###

Logic: The agent has to reproduce to two angles separated by a constant delay.

Reference paper: 

[Visual perception as retrospective Bayesian
        decoding from high- to low-level features](https://www.pnas.org/content/114/43/E9115.short)

Default Epoch timing (ms) 

fixation : constant 500

stim1 : constant 500

delay1 : constant 500

stim2 : constant 500

delay2 : constant 500

go1 : constant 500

go2 : constant 500

### AntiReach1D task ###

Logic: The agent has to move in the direction opposite to the one indicated by the observation.

Reference paper: 

[Look away: the anti-saccade task and
        the voluntary control of eye movement](https://www.nature.com/articles/nrn1345)

Default Epoch timing (ms) 

fixation : constant 500

reach : constant 500

### Bandit task ###

Logic: The agent has to select between N actions with different reward probabilities.

Reference paper: 

[Prefrontal cortex as a meta-reinforcement learning system](https://www.nature.com/articles/s41593-018-0147-8)

Other parameters: 

probs : Reward probabilities for each arm. (def: (.9, .1))

n_arms : Number of arms. (def: 2)

### CVLearning task ###

Logic: Implements shaping for the delay-response task, in which agents have to integrate two stimuli and report which one is larger on average after a delay.

Reference paper: 

[Discrete attractor dynamics underlies persistent activity in the frontal cortex](https://www.nature.com/articles/s41586-019-0919-7)

Default Epoch timing (ms) 

fixation : constant 200

stimulus : constant 1150

delay : choice [300, 500, 700, 900, 1200, 2000, 3200, 4000]

decision : constant 1500

Other parameters: 

stimEv : Controls the difficulty of the experiment. (def: 1.)

### DPA task ###

Logic: A sample is followed by a delay and a test. Agents have to report if the pair sample-test is a rewarded pair or not.

Reference paper: 

[Active information maintenance in working memory by a sensory cortex](https://elifesciences.org/articles/43191)

Default Epoch timing (ms) 

fixation : constant 0

stim1 : constant 1000

delay_btw_stim : constant 13000

stim2 : constant 1000

delay_aft_stim : constant 1000

decision : constant 500

Other parameters: 

noise : Standard deviation of the Gaussian noise added to
        the stimulus. (def: 0.01)

### DawTwoStep task ###

Logic: On each trial, an initial choice between two options lead to either of two, second-stage states. In turn, these both demand another two-option choice, each of which is associated with  a different chance of receiving reward.

Reference paper: 

[Model-Based Influences on Humans'
        Choices and Striatal Prediction Errors](https://www.sciencedirect.com/science/article/pii/S0896627311001255)

### DelayedMatchToSample task ###

Logic: A sample stimulus is followed by a delay and test. Agents are required to indicate if the sample and test are the same stimulus.

Reference paper: 

[Neural Mechanisms of Visual Working Memory
        in Prefrontal Cortex of the Macaque](https://www.jneurosci.org/content/jneuro/16/16/5154.full.pdf)

Default Epoch timing (ms) 

fixation : constant 300

sample : constant 500

delay : constant 1000

test : constant 500

decision : constant 900

### DelayedMatchToSampleDistractor1D task ###

Logic: Delay Match to sample with multiple,
         potentially repeating distractors.

Reference paper: 

[Neural Mechanisms of Visual Working Memory
        in Prefrontal Cortex of the Macaque](https://www.jneurosci.org/content/jneuro/16/16/5154.full.pdf)

Default Epoch timing (ms) 

fixation : constant 300

sample : constant 500

delay1 : constant 1000

test1 : constant 500

delay2 : constant 1000

test2 : constant 500

delay3 : constant 1000

test3 : constant 500

### DR task ###

Logic: Agents have to integrate two stimuli and report which one is larger on average after a delay.

Reference paper: 

[Discrete attractor dynamics underlies persistent activity in the frontal cortex](https://www.nature.com/articles/s41586-019-0919-7)

Default Epoch timing (ms) 

fixation : constant 0

stimulus : constant 1150

delay : choice [300, 500, 700, 900, 1200, 2000, 3200, 4000]

decision : constant 1500

Other parameters: 

stimEv : Controls the difficulty of the experiment. (def: 1.)

### Detection task ###

Logic: The agent has to GO if a stimulus is presented.

Reference paper: 

Missing paper name

Missing paper link

Default Epoch timing (ms) 

fixation : constant 500

stimulus : truncated_exponential [1000, 500, 1500]

Other parameters: 

stim_dur : Stimulus duration. (def: 100, ms)

delay : If not None indicates the delay, from the moment of the start of the stimulus period when the actual stimulus is presented. Otherwise, the delay is drawn from a uniform distribution. (def: None)

noise : Standard deviation of background noise. (def: 1)

### GNG task ###

Logic: Go/No-Go task in which the subject has either Go (e.g. lick) or not Go depending on which one of two stimuli is presented with.

Reference paper: 

[Active information maintenance in working memory by a sensory cortex](https://elifesciences.org/articles/43191)

Default Epoch timing (ms) 

fixation : constant 0

stimulus : constant 500

resp_delay : constant 500

decision : constant 500

### IntervalDiscrimination task ###

Logic: Agents have to report which of two stimuli presented sequentially is longer.

Reference paper: 

[Feature- and Order-Based Timing Representations
         in the Frontal Cortex](https://www.sciencedirect.com/science/article/pii/S0896627309004887)

Default Epoch timing (ms) 

fixation : constant 300

stim1 : uniform (300, 600)

delay1 : choice [800, 1500]

stim2 : uniform (300, 600)

delay2 : constant 500

decision : constant 300

### Mante task ###

Logic: Agent has to perform one of two different perceptual discriminations. On every trial, a contextual cue indicates which one to perform.

Reference paper: 

[Context-dependent computation by recurrent
         dynamics in prefrontal cortex](https://www.nature.com/articles/nature12742)

Default Epoch timing (ms) 

fixation : constant 300

stimulus : constant 750

delay : truncated_exponential [600, 300, 3000]

decision : constant 100

### MatchingPenny task ###

Logic: The agent is rewarded when it selects the
         same target as the computer.

Reference paper: 

[Prefrontal cortex and decision making in a
         mixed-strategy game](https://www.nature.com/articles/nn1209)

Other parameters: 

opponent_type : Type of opponent. (def: 'random')

### MotorTiming task ###

Logic: Agents have to produce different time
         intervals using different effectors (actions).

Reference paper: 

[Flexible timing by temporal scaling of
         cortical responses](https://www.nature.com/articles/s41593-017-0028-6)

Default Epoch timing (ms) 

fixation : constant 500

cue : uniform [1000, 3000]

set : constant 50

### nalt_RDM task ###

Logic: N-alternative forced choice task in which the subject
         has to integrate N stimuli to decide which one is higher
          on average.

Reference paper: 

Missing paper name

Missing paper link

Default Epoch timing (ms) 

fixation : constant 500

stimulus : truncated_exponential [330, 80, 1500]

decision : constant 500

Other parameters: 

n_ch : Number of choices. (def: 3)

stimEv : Controls the difficulty of the experiment. (def: 1.)

### RDM task ###

Logic: Random dot motion task. Two-alternative forced
         choice task in which the subject has to integrate two stimuli to
         decide which one is higher on average.

Reference paper: 

[The analysis of visual motion: a comparison of
        neuronal and psychophysical performance](https://www.jneurosci.org/content/12/12/4745)

Default Epoch timing (ms) 

fixation : constant 100

stimulus : constant 2000

decision : constant 100

Other parameters: 

stimEv : Controls the difficulty of the experiment. (def: 1.)

### Reaching1D task ###

Logic: The agent has to reproduce the angle indicated
         by the observation.

Reference paper: 

[Neuronal population coding of movement direction](https://science.sciencemag.org/content/233/4771/1416)

Default Epoch timing (ms) 

fixation : constant 500

reach : constant 500

### Reaching1DWithSelfDistraction task ###

Logic: The agent has to reproduce the angle indicated
         by the observation. Furthermore, the reaching state itself
         generates strong inputs that overshadows the actual target input.

Reference paper: 

Missing paper name

Missing paper link

Default Epoch timing (ms) 

fixation : constant 500

reach : constant 500

### ReadySetGo task ###

Logic: Agents have to measure and produce different time
         intervals.

Reference paper: 

[Flexible Sensorimotor Computations through Rapid
        Reconfiguration of Cortical Dynamics](https://www.sciencedirect.com/science/article/pii/S0896627318304185)

Default Epoch timing (ms) 

fixation : constant 100

ready : constant 83

measure : choice [800, 1500]

set : constant 83

Other parameters: 

gain : Controls the measure that the agent has to produce. (def: 1)

### Romo task ###

Logic: Two-alternative forced choice task in which
         the subject has to compare two stimuli separated by a delay
         to decide which one has a higher frequency.

Reference paper: 

[Neuronal Population Coding of Parametric
        Working Memory](https://www.jneurosci.org/content/30/28/9424)

Default Epoch timing (ms) 

fixation : uniform (1500, 3000)

f1 : constant 500

delay : constant 3000

f2 : constant 500

decision : constant 100

### Serrano task ###

Logic: Missing description

Reference paper: 

Missing paper name

Missing paper link

Default Epoch timing (ms) 

stimulus : constant 100

delay : choice [0, 100, 200]

decision : constant 300

### PadoaSch task ###

Logic: Agents choose between two stimuli (A and B; where A
         is preferred) offered in different amounts.

Reference paper: 

[Neurons in the orbitofrontal cortex encode
         economic value](https://www.nature.com/articles/nature04676)

Default Epoch timing (ms) 

fixation : constant 1500

offer_on : uniform [1000, 2000]

decision : constant 750

### PDWager task ###

Logic: Agents do a discrimination task (see RDM). On a
         random half of the trials, the agent is given the option to abort
         the direction discrimination and to choose instead a small but
         certain reward associated with a action.

Reference paper: 

[Representation of Confidence Associated with a
         Decision by Neurons in the Parietal Cortex](https://science.sciencemag.org/content/324/5928/759.long)

Default Epoch timing (ms) 

fixation : constant 100

stimulus : truncated_exponential [180, 100, 900]

delay : truncated_exponential [1350, 1200, 1800]

pre_sure : uniform [500, 750]

decision : constant 100






### List of wrappers implemented

* 9 wrappers implemented so far.

### CatchTrials-v0 wrapper ###

Logic: Introduces catch trials in which the reward for
         a correct choice is modified (e.g. is set to the reward for an
         incorrect choice). Note that the wrapper only changes the reward
         associated to a correct answer and does not change the ground truth.
         Thus, the catch trial affect a pure supervised learning setting.

Input parameters: 

stim_th : Percentile of stimulus distribution below which catch
        trials are allowed (in some cases, experimenter might decide not
        to have catch trials when  stimulus is very obvious). (def: 50)

catch_prob : Catch trial probability. (def: 0.1)

start : Number of trials after which the catch trials can occur.
        (def: 0)

### MissTrialReward-v0 wrapper ###

Logic: Add a negative reward if a trial ends with no action.

Input parameters: 

r_miss : Reward given when a miss trial occurs.(def: 0)

### Monitor-v0 wrapper ###

Logic: Saves relevant behavioral information: rewards, actions, observations, new trial, ground truth.

Input parameters: 

folder : Folder where the data will be saved. (def: None)

num_tr_save : Data will be saved every num_tr_save trials.
        (def: 100000)

info_keywords : (tuple) extra information to log, from the information return of environment.step

verbose : Whether to print information about average reward and number of trials

### Noise-v0 wrapper ###

Logic: Add Gaussian noise to the observations.

Input parameters: 

std_noise : Standard deviation of noise. (def: 0.1)

### PassAction-v0 wrapper ###

Logic: Modifies observation by adding the previous
        action.

### PassReward-v0 wrapper ###

Logic: Modifies observation by adding the previous
        reward.

### ReactionTime-v0 wrapper ###

Logic: Modfies a given environment by allowing the network
        to act at any time after the fixation period.

### SideBias-v0 wrapper ###

Logic: Changes the probability of ground truth.

Input parameters: 

prob : Specifies probabilities for each choice. Within each block,
        the probability should sum up to 1.
        (def: None (Numpy array (n_block, n_choices)))

block_dur : Number of trials per block. (def: 200 (int))

### TrialHistory-v0 wrapper ###

Logic: Change ground truth probability based on previous outcome.

Reference paper: 

[Response outcomes gate the impact of expectations
         on perceptual decisions](https://www.biorxiv.org/content/10.1101/433409v3)

Input parameters: 

blk_ch_prob : If not None, specifies the probability of changing
        block (randomly). (def: None)

rep_prob : Specifies probabilities of repeating for each block.
        (def: (.2, .8))

block_dur : Number of trials per block. (def: 200 (int))

