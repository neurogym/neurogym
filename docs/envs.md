* Under development, details subject to change

### List of tasks implemented

* 26 tasks implemented so far.

[AngleReproduction-v0](#anglereproduction)

[AntiReach-v0](#antireach1d)

[Bandit-v0](#bandit)

[CVLearning-v0](#cvlearning)

[ChangingEnvironment-v0](#changingenvironment)

[DawTwoStep-v0](#dawtwostep)

[DelayPairedAssociation-v0](#delaypairedassociation)

[DelayedMatchCategory-v0](#delayedmatchcategory)

[DelayedMatchSample-v0](#delayedmatchtosample)

[DelayedMatchToSampleDistractor1D-v0](#delayedmatchtosampledistractor1d)

[Detection-v0](#detection)

[GoNogo-v0](#gonogo)

[IntervalDiscrimination-v0](#intervaldiscrimination)

[Mante-v0](#mante)

[MatchingPenny-v0](#matchingpenny)

[MotorTiming-v0](#motortiming)

[NAltPerceptualDecisionMaking-v0](#nalt_perceptualdecisionmaking)

[PerceptualDecisionMaking-v0](#perceptualdecisionmaking)

[PerceptualDecisionMakingDelayResponse-v0](#perceptualdecisionmakingdelayresponse)

[PostDecisionWager-v0](#postdecisionwager)

[Reaching1D-v0](#reaching1d)

[Reaching1DWithSelfDistraction-v0](#reaching1dwithselfdistraction)

[ReachingDelayResponse-v0](#reachingdelayresponse)

[ReadySetGo-v0](#readysetgo)

[Romo-v0](#romo)

[padoaSch-v0](#padoasch)

___

### AngleReproduction

Logic: The agent has to reproduce to two angles separated by a constant delay.

Reference paper: 

[Visual perception as retrospective Bayesian decoding from high- to low-level features](https://www.pnas.org/content/114/43/E9115.short)

Default Period timing (ms) 

fixation : constant 500

stim1 : constant 500

delay1 : constant 500

stim2 : constant 500

delay2 : constant 500

go1 : constant 500

go2 : constant 500

Tags: perceptual, working memory, delayed response, steps action space.

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/anglereproduction.py)

___

### AntiReach1D

Logic: The agent has to move in the direction opposite to the one indicated by the observation.

Reference paper: 

[Look away: the anti-saccade task and the voluntary control of eye movement](https://www.nature.com/articles/nrn1345)

Default Period timing (ms) 

fixation : constant 500

reach : constant 500

Tags: perceptual, steps action space.

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/antireach.py)

___

### Bandit

Logic: The agent has to select between N actions with different reward probabilities.

Reference paper: 

[Prefrontal cortex as a meta-reinforcement learning system](https://www.nature.com/articles/s41593-018-0147-8)

Other parameters: 

n_arms : Number of arms. (def: 2)

gt_arm : High reward arm. (def: 0)

probs : Reward probabilities for each arm. (def: (.9, .1))

Tags: n-alternative, supervised.

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/bandit.py)

___

### CVLearning

Logic: Implements shaping for the delay-response task, in which agents have to integrate two stimuli and report which one is larger on average after a delay.

Reference paper: 

[Discrete attractor dynamics underlies persistent activity in the frontal cortex](https://www.nature.com/articles/s41586-019-0919-7)

Default Period timing (ms) 

fixation : constant 200

stimulus : constant 1150

delay : choice [300, 500, 700, 900, 1200, 2000, 3200, 4000]

decision : constant 1500

Other parameters: 

stimEv : Controls the difficulty of the experiment. (def: 1.)

Tags: perceptual, delayed response, two-alternative, supervised.

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/cv_learning.py)

___

### ChangingEnvironment

Logic: Random Dots Motion tasks in which the correct action depends on a randomly changing context

Reference paper: 

[Hierarchical decision processes that operate over distinct timescales underlie choice and changes in strategy](https://www.pnas.org/content/113/31/E4531)

Default Period timing (ms) 

fixation : constant 500

stimulus : truncated_exponential [1000, 500, 1500]

decision : constant 500

Other parameters: 

cxt_ch_prob : Probability of changing context.

cxt_cue : Whether to show context as a cue.

stimEv : Controls the difficulty of the experiment. (def: 1.)

Tags: perceptual, 2-alternative, supervised, context dependent.

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/changingenvironment.py)

___

### DawTwoStep

Logic: On each trial, an initial choice between two options lead to either of two, second-stage states. In turn, these both demand another two-option choice, each of which is associated with a different chance of receiving reward.

Reference paper: 

[Model-Based Influences on Humans Choices and Striatal Prediction Errors](https://www.sciencedirect.com/science/article/pii/S0896627311001255)

Tags: two-alternative, supervised.

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/dawtwostep.py)

___

### DelayPairedAssociation

Logic: A sample is followed by a delay and a test. Agents have to report if the pair sample-test is a rewarded pair or not.

Reference paper: 

[Active information maintenance in working memory by a sensory cortex](https://elifesciences.org/articles/43191)

Default Period timing (ms) 

fixation : constant 0

stim1 : constant 1000

delay_btw_stim : constant 13000

stim2 : constant 1000

delay_aft_stim : constant 1000

decision : constant 500

Other parameters: 

noise : Standard deviation of the Gaussian noise added to the stimulus. (def: 0.01)

Tags: perceptual, working memory, go/no-go, supervised.

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/delaypairedassociation.py)

___

### DelayedMatchCategory

Logic: A sample stimulus is followed by a delay and test. Agents are required to indicate if the sample and test are in the same category.

Reference paper: 

[Experience-dependent representation of visual categories in parietal cortex](https://www.nature.com/articles/nature05078)

Default Period timing (ms) 

fixation : constant 500

sample : constant 650

first_delay : constant 1000

test : constant 650

Tags: perceptual, working memory, two-alternative, supervised.

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/delaymatchcategory.py)

___

### DelayedMatchToSample

Logic: A sample stimulus is followed by a delay and test. Agents are required to indicate if the sample and test are the same stimulus.

Reference paper: 

[Neural Mechanisms of Visual Working Memory in Prefrontal Cortex of the Macaque](https://www.jneurosci.org/content/jneuro/16/16/5154.full.pdf)

Default Period timing (ms) 

fixation : constant 300

sample : constant 500

delay : constant 1000

test : constant 500

decision : constant 900

Tags: perceptual, working memory, two-alternative, supervised.

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/delaymatchsample.py)

___

### DelayedMatchToSampleDistractor1D

Logic: Delay Match to sample with multiple, potentially repeating distractors.

Reference paper: 

[Neural Mechanisms of Visual Working Memory in Prefrontal Cortex of the Macaque](https://www.jneurosci.org/content/jneuro/16/16/5154.full.pdf)

Default Period timing (ms) 

fixation : constant 300

sample : constant 500

delay1 : constant 1000

test1 : constant 500

delay2 : constant 1000

test2 : constant 500

delay3 : constant 1000

test3 : constant 500

Tags: perceptual, working memory, two-alternative, supervised.

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/delaymatchsample.py)

___

### Detection

Logic: The agent has to GO if a stimulus is presented.

Reference paper: 

Missing paper name

Missing paper link

Default Period timing (ms) 

fixation : constant 500

stimulus : truncated_exponential [1000, 500, 1500]

Other parameters: 

noise : Standard deviation of background noise. (def: 1)

delay : If not None indicates the delay, from the moment of the start of the stimulus period when the actual stimulus is presented. Otherwise, the delay is drawn from a uniform distribution. (def: None)

stim_dur : Stimulus duration. (def: 100, ms)

Tags: perceptual, reaction time, go/no-go, supervised.

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/detection.py)

___

### GoNogo

Logic: Go/No-Go task in which the subject has either Go (e.g. lick) or not Go depending on which one of two stimuli is presented with.

Reference paper: 

[Active information maintenance in working memory by a sensory cortex](https://elifesciences.org/articles/43191)

Default Period timing (ms) 

fixation : constant 0

stimulus : constant 500

resp_delay : constant 500

decision : constant 500

Tags: delayed response, go/no-go, supervised.

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/gonogo.py)

___

### IntervalDiscrimination

Logic: Agents have to report which of two stimuli presented sequentially is longer.

Reference paper: 

[Feature- and Order-Based Timing Representations in the Frontal Cortex](https://www.sciencedirect.com/science/article/pii/S0896627309004887)

Default Period timing (ms) 

fixation : constant 300

stim1 : uniform (300, 600)

delay1 : choice [800, 1500]

stim2 : uniform (300, 600)

delay2 : constant 500

decision : constant 300

Tags: timing, working memory, delayed response, two-alternative, supervised.

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/intervaldiscrimination.py)

___

### Mante

Logic: Agent has to perform one of two different perceptual discriminations. On every trial, a contextual cue indicates which one to perform.

Reference paper: 

[Context-dependent computation by recurrent dynamics in prefrontal cortex](https://www.nature.com/articles/nature12742)

Default Period timing (ms) 

fixation : constant 300

stimulus : constant 750

delay : truncated_exponential [600, 300, 3000]

decision : constant 100

Tags: perceptual, context dependent, two-alternative, supervised.

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/mante.py)

___

### MatchingPenny

Logic: The agent is rewarded when it selects the same target as the computer.

Reference paper: 

[Prefrontal cortex and decision making in a mixed-strategy game](https://www.nature.com/articles/nn1209)

Other parameters: 

opponent_type : Type of opponent. (def: 'random')

Tags: two-alternative, supervised.

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/matchingpenny.py)

___

### MotorTiming

Logic: Agents have to produce different time intervals using different effectors (actions). [different actions not implemented]

Reference paper: 

[Flexible timing by temporal scaling of cortical responses](https://www.nature.com/articles/s41593-017-0028-6)

Default Period timing (ms) 

fixation : constant 500

cue : uniform [1000, 3000]

set : constant 50

Tags: timing, go/no-go, supervised.

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/readysetgo.py)

___

### nalt_PerceptualDecisionMaking

Logic: N-alternative forced choice task in which the subject has to integrate N stimuli to decide which one is higher on average.

Reference paper: 

Missing paper name

Missing paper link

Default Period timing (ms) 

fixation : constant 500

stimulus : truncated_exponential [330, 80, 1500]

decision : constant 500

Other parameters: 

stimEv : Controls the difficulty of the experiment. (def: 1.)

n_ch : Number of choices. (def: 3)

Tags: perceptual, n-alternative, supervised.

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/nalt_perceptualdecisionmaking.py)

___

### PerceptualDecisionMaking

Logic: Random dot motion task. Two-alternative forced choice task in which the subject has to integrate two stimuli to decide which one is higher on average.

Reference paper: 

[The analysis of visual motion: a comparison of neuronal and psychophysical performance](https://www.jneurosci.org/content/12/12/4745)

Default Period timing (ms) 

fixation : constant 100

stimulus : constant 2000

decision : constant 100

Other parameters: 

stimEv : Controls the difficulty of the experiment. (def: 1.)

Tags: perceptual, 2-alternative, supervised.

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/perceptualdecisionmaking.py)

___

### PerceptualDecisionMakingDelayResponse

Logic: Agents have to integrate two stimuli and report which one is larger on average after a delay.

Reference paper: 

[Discrete attractor dynamics underlies persistent activity in the frontal cortex](https://www.nature.com/articles/s41586-019-0919-7)

Default Period timing (ms) 

fixation : constant 0

stimulus : constant 1150

delay : choice [300, 500, 700, 900, 1200, 2000, 3200, 4000]

decision : constant 1500

Other parameters: 

stimEv : Controls the difficulty of the experiment. (def: 1.)

Tags: perceptual, delayed response, two-alternative, supervised.

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/perceptualdecisionmaking.py)

___

### PostDecisionWager

Logic: Agents do a discrimination task (see PerceptualDecisionMaking). On a random half of the trials, the agent is given the option to abort the direction discrimination and to choose instead a small but certain reward associated with a action.

Reference paper: 

[Representation of Confidence Associated with a Decision by Neurons in the Parietal Cortex](https://science.sciencemag.org/content/324/5928/759.long)

Default Period timing (ms) 

fixation : constant 100

stimulus : truncated_exponential [180, 100, 900]

delay : truncated_exponential [1350, 1200, 1800]

pre_sure : uniform [500, 750]

decision : constant 100

Tags: perceptual, delayed response, confidence, supervised.

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/postdecisionwager.py)

___

### Reaching1D

Logic: The agent has to reproduce the angle indicated by the observation.

Reference paper: 

[Neuronal population coding of movement direction](https://science.sciencemag.org/content/233/4771/1416)

Default Period timing (ms) 

fixation : constant 500

reach : constant 500

Tags: motor, steps action space.

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/reaching.py)

___

### Reaching1DWithSelfDistraction

Logic: The agent has to reproduce the angle indicated by the observation. Furthermore, the reaching state itself generates strong inputs that overshadows the actual target input.

Reference paper: 

Missing paper name

Missing paper link

Default Period timing (ms) 

fixation : constant 500

reach : constant 500

Tags: motor, steps action space.

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/reaching.py)

___

### ReachingDelayResponse

Logic: Working memory visual spatial task ~ Funahashi et al. 1991 adapted to freely moving mice in a continous choice-space.Brief description: while fixating, stimulus is presented in a touchscreen (bright circle). Afterwards (perhaps including an extra delay), doors open allowing the mouse to touch the screen where the stimulus was located.

Reference paper: 

Missing paper name

Missing paper link

Default Period timing (ms) 

stimulus : constant 500

delay : choice [0, 1000, 2000]

decision : constant 5000

Tags: perceptual, delayed response, continuous action space, multidimensional action space, supervised.

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/reachingdelayresponse.py)

___

### ReadySetGo

Logic: Agents have to measure and produce different time intervals.

Reference paper: 

[Flexible Sensorimotor Computations through Rapid Reconfiguration of Cortical Dynamics](https://www.sciencedirect.com/science/article/pii/S0896627318304185)

Default Period timing (ms) 

fixation : constant 100

ready : constant 83

measure : choice [800, 1500]

set : constant 83

Other parameters: 

gain : Controls the measure that the agent has to produce. (def: 1)

Tags: timing, go/no-go, supervised.

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/readysetgo.py)

___

### Romo

Logic: Two-alternative forced choice task in which the subject has to compare two stimuli separated by a delay to decide which one has a higher frequency.

Reference paper: 

[Neuronal Population Coding of Parametric Working Memory](https://www.jneurosci.org/content/30/28/9424)

Default Period timing (ms) 

fixation : uniform (1500, 3000)

f1 : constant 500

delay : constant 3000

f2 : constant 500

decision : constant 100

Tags: perceptual, working memory, two-alternative, supervised.

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/romo.py)

___

### PadoaSch

Logic: Agents choose between two stimuli (A and B; where A is preferred) offered in different amounts.

Reference paper: 

[Neurons in the orbitofrontal cortex encode economic value](https://www.nature.com/articles/nature04676)

Default Period timing (ms) 

fixation : constant 1500

offer_on : uniform [1000, 2000]

decision : constant 750

Tags: perceptual, value-based.

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/padoa_sch.py)

___

### Tags ### 

### 2-alternative 

[PerceptualDecisionMaking-v0](perceptualdecisionmaking)

[ChangingEnvironment-v0](changingenvironment)

### confidence 

[PostDecisionWager-v0](postdecisionwager)

### context dependent 

[Mante-v0](mante)

[ChangingEnvironment-v0](changingenvironment)

### continuous action space 

[ReachingDelayResponse-v0](reachingdelayresponse)

### delayed response 

[PostDecisionWager-v0](postdecisionwager)

[GoNogo-v0](gonogo)

[PerceptualDecisionMakingDelayResponse-v0](perceptualdecisionmakingdelayresponse)

[IntervalDiscrimination-v0](intervaldiscrimination)

[AngleReproduction-v0](anglereproduction)

[ReachingDelayResponse-v0](reachingdelayresponse)

[CVLearning-v0](cvlearning)

### go/no-go 

[DelayPairedAssociation-v0](delaypairedassociation)

[GoNogo-v0](gonogo)

[ReadySetGo-v0](readysetgo)

[MotorTiming-v0](motortiming)

[Detection-v0](detection)

### motor 

[Reaching1D-v0](reaching1d)

[Reaching1DWithSelfDistraction-v0](reaching1dwithselfdistraction)

### multidimensional action space 

[ReachingDelayResponse-v0](reachingdelayresponse)

### n-alternative 

[Bandit-v0](bandit)

[NAltPerceptualDecisionMaking-v0](nalt_perceptualdecisionmaking)

### perceptual 

[Mante-v0](mante)

[Romo-v0](romo)

[PerceptualDecisionMaking-v0](perceptualdecisionmaking)

[padoaSch-v0](padoasch)

[PostDecisionWager-v0](postdecisionwager)

[DelayPairedAssociation-v0](delaypairedassociation)

[DelayedMatchSample-v0](delayedmatchtosample)

[DelayedMatchCategory-v0](delayedmatchcategory)

[PerceptualDecisionMakingDelayResponse-v0](perceptualdecisionmakingdelayresponse)

[NAltPerceptualDecisionMaking-v0](nalt_perceptualdecisionmaking)

[AntiReach-v0](antireach1d)

[DelayedMatchToSampleDistractor1D-v0](delayedmatchtosampledistractor1d)

[AngleReproduction-v0](anglereproduction)

[Detection-v0](detection)

[ReachingDelayResponse-v0](reachingdelayresponse)

[CVLearning-v0](cvlearning)

[ChangingEnvironment-v0](changingenvironment)

### reaction time 

[Detection-v0](detection)

### steps action space 

[Reaching1D-v0](reaching1d)

[Reaching1DWithSelfDistraction-v0](reaching1dwithselfdistraction)

[AntiReach-v0](antireach1d)

[AngleReproduction-v0](anglereproduction)

### supervised 

[Mante-v0](mante)

[Romo-v0](romo)

[PerceptualDecisionMaking-v0](perceptualdecisionmaking)

[PostDecisionWager-v0](postdecisionwager)

[DelayPairedAssociation-v0](delaypairedassociation)

[GoNogo-v0](gonogo)

[ReadySetGo-v0](readysetgo)

[DelayedMatchSample-v0](delayedmatchtosample)

[DelayedMatchCategory-v0](delayedmatchcategory)

[DawTwoStep-v0](dawtwostep)

[MatchingPenny-v0](matchingpenny)

[MotorTiming-v0](motortiming)

[Bandit-v0](bandit)

[PerceptualDecisionMakingDelayResponse-v0](perceptualdecisionmakingdelayresponse)

[NAltPerceptualDecisionMaking-v0](nalt_perceptualdecisionmaking)

[DelayedMatchToSampleDistractor1D-v0](delayedmatchtosampledistractor1d)

[IntervalDiscrimination-v0](intervaldiscrimination)

[Detection-v0](detection)

[ReachingDelayResponse-v0](reachingdelayresponse)

[CVLearning-v0](cvlearning)

[ChangingEnvironment-v0](changingenvironment)

### timing 

[ReadySetGo-v0](readysetgo)

[MotorTiming-v0](motortiming)

[IntervalDiscrimination-v0](intervaldiscrimination)

### two-alternative 

[Mante-v0](mante)

[Romo-v0](romo)

[DelayedMatchSample-v0](delayedmatchtosample)

[DelayedMatchCategory-v0](delayedmatchcategory)

[DawTwoStep-v0](dawtwostep)

[MatchingPenny-v0](matchingpenny)

[PerceptualDecisionMakingDelayResponse-v0](perceptualdecisionmakingdelayresponse)

[DelayedMatchToSampleDistractor1D-v0](delayedmatchtosampledistractor1d)

[IntervalDiscrimination-v0](intervaldiscrimination)

[CVLearning-v0](cvlearning)

### value-based 

[padoaSch-v0](padoasch)

### working memory 

[Romo-v0](romo)

[DelayPairedAssociation-v0](delaypairedassociation)

[DelayedMatchSample-v0](delayedmatchtosample)

[DelayedMatchCategory-v0](delayedmatchcategory)

[DelayedMatchToSampleDistractor1D-v0](delayedmatchtosampledistractor1d)

[IntervalDiscrimination-v0](intervaldiscrimination)

[AngleReproduction-v0](anglereproduction)

