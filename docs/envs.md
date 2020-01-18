### List of environments implemented

* 24 tasks implemented so far.

* Under development, details subject to change

#### AngleReproduction

Missing description

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

#### AntiReach1D

Missing description

Reference paper: 

[Look away: the anti-saccade task and 
        the voluntary control of eye movement](https://www.nature.com/articles/nrn1345)

Default Epoch timing (ms) 

fixation : constant 500

reach : constant 500

#### Bandit

requires the agent to select between N actions with different reward probabilities.

Reference paper: 

[Prefrontal cortex as a meta-reinforcement learning system](https://www.nature.com/articles/s41593-018-0147-8)

#### DPA

A sample is followed by a delay and a test. Agents have to report if the pair sample-test is a rewarded pair or not.

Reference paper: 

[Active information maintenance in working memory by a sensory cortex](https://elifesciences.org/articles/43191)

Default Epoch timing (ms) 

fixation : constant 0

stim1 : constant 1000

delay_btw_stim : constant 13000

stim2 : constant 1000

delay_aft_stim : constant 1000

decision : constant 500

#### DawTwoStep

Missing description

Reference paper: 

[Model-Based Influences on Humans' 
        Choices and Striatal Prediction Errors](https://www.sciencedirect.com/science/article/pii/S0896627311001255)

#### DelayedMatchCategory

A sample stimulus is followed by a delay and test. Agents are required to indicate if the sample and test are in the same category.

Reference paper: 

[Experience-dependent representation
        of visual categories in parietal cortex](https://www.nature.com/articles/nature05078)

Default Epoch timing (ms) 

fixation : constant 500

sample : constant 650

first_delay : constant 1000

test : constant 650

second_delay : constant 250

decision : constant 650

#### DelayedMatchToSample

A sample stimulus is followed by a delay and test. Agents are required to indicate if the sample and test are the same stimulus.

Reference paper: 

[Neural Mechanisms of Visual Working Memory 
        in Prefrontal Cortex of the Macaque](https://www.jneurosci.org/content/jneuro/16/16/5154.full.pdf)

Default Epoch timing (ms) 

fixation : constant 300

sample : constant 500

delay : constant 1000

test : constant 500

decision : constant 900

#### DelayedMatchToSampleDistractor1D

Missing description

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

#### DR

Agents have to integrate two stimuli and report which one is larger on average after a delay.

Reference paper: 

[Discrete attractor dynamics underlies persistent activity in the frontal cortex](https://www.nature.com/articles/s41586-019-0919-7)

Default Epoch timing (ms) 

fixation : constant 0

stimulus : constant 1150

delay : choice [300, 500, 700, 900, 1200, 2000, 3200, 4000]

go_cue : constant 100

decision : constant 1500

#### GNG

Go/No-Go task in which the subject has either Go (e.g. lick) or not Go depending on which one of two stimuli is presented with

Reference paper: 

[Active information maintenance in working memory by a sensory cortex](https://elifesciences.org/articles/43191)

Default Epoch timing (ms) 

fixation : constant 0

stimulus : constant 500

resp_delay : constant 500

decision : constant 500

#### GenTask

Missing description

Reference paper: 

Missing paper name

Missing paper link

Default Epoch timing (ms) 

fixation : truncated_exponential [500, 200, 800]

stim1 : truncated_exponential [500, 200, 800]

delay_btw_stim : truncated_exponential [500, 200, 800]

stim2 : truncated_exponential [500, 200, 800]

delay_aft_stim : truncated_exponential [500, 200, 800]

decision : truncated_exponential [500, 200, 800]

#### IntervalDiscrimination

Agents have to report which of two stimuli presented sequentially is longer.

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

#### Mante

Missing description

Reference paper: 

[Context-dependent computation by recurrent 
        dynamics in prefrontal cortex](https://www.nature.com/articles/nature12742)

Default Epoch timing (ms) 

fixation : constant 300

target : constant 350

stimulus : constant 750

delay : truncated_exponential [600, 300, 3000]

decision : constant 100

#### MatchingPenny

Missing description

Reference paper: 

[Prefrontal cortex and decision making in a mixed-strategy game](https://www.nature.com/articles/nn1209)

#### MemoryRecall

Missing description

Reference paper: 

Missing paper name

Missing paper link

#### MotorTiming

Missing description

Reference paper: 

[Flexible timing by temporal scaling of cortical responses](https://www.nature.com/articles/s41593-017-0028-6)

Default Epoch timing (ms) 

fixation : constant 500

cue : uniform [1000, 3000]

set : constant 50

#### nalt_RDM

N-alternative forced choice task in which the subject has to integrate N stimuli to decide which one is higher on average

Reference paper: 

Missing paper name

Missing paper link

Default Epoch timing (ms) 

fixation : constant 500

stimulus : truncated_exponential [330, 80, 1500]

decision : constant 500

#### RDM

Random dot motion task. Two-alternative forced choice task in which the subject has to integrate two stimuli to decide which one is higher on average

Reference paper: 

[The analysis of visual motion: a comparison of
        neuronal and psychophysical performance](https://www.jneurosci.org/content/12/12/4745)

Default Epoch timing (ms) 

fixation : constant 100

stimulus : constant 2000

decision : constant 100

#### Reaching1D

Missing description

Reference paper: 

[Neuronal population coding of movement direction](https://science.sciencemag.org/content/233/4771/1416)

Default Epoch timing (ms) 

fixation : constant 500

reach : constant 500

#### Reaching1DWithSelfDistraction

Missing description

Reference paper: 

Missing paper name

Missing paper link

Default Epoch timing (ms) 

fixation : constant 500

reach : constant 500

#### ReadySetGo

Missing description

Reference paper: 

[Flexible Sensorimotor Computations through Rapid
        Reconfiguration of Cortical Dynamics](https://www.sciencedirect.com/science/article/pii/S0896627318304185)

Default Epoch timing (ms) 

fixation : constant 100

ready : constant 83

measure : choice [800, 1500]

set : constant 83

#### Romo

Missing description

Reference paper: 

[Neuronal Population Coding of Parametric
        Working Memory](https://www.jneurosci.org/content/30/28/9424)

Default Epoch timing (ms) 

fixation : uniform (1500, 3000)

f1 : constant 500

delay : constant 3000

f2 : constant 500

decision : constant 100

#### PadoaSch

Missing description

Reference paper: 

[Neurons in the orbitofrontal cortex encode economic value](https://www.nature.com/articles/nature04676)

Default Epoch timing (ms) 

fixation : constant 1500

offer_on : uniform [1000, 2000]

decision : constant 750

#### PDWager

Missing description

Reference paper: 

[Representation of Confidence Associated with a
         Decision by Neurons in the Parietal Cortex](https://science.sciencemag.org/content/324/5928/759.long)

Default Epoch timing (ms) 

fixation : constant 100

target : constant 0

stimulus : truncated_exponential [180, 100, 900]

delay : truncated_exponential [1350, 1200, 1800]

pre_sure : uniform [500, 750]

decision : constant 100

