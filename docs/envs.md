* Under development, details subject to change

### List of 26 tasks implemented

[AngleReproduction-v0](#anglereproduction)

[AntiReach-v0](#antireach1d)

[Bandit-v0](#bandit)

[CVLearning-v0](#cvlearning)

[ChangingEnvironment-v0](#changingenvironment)

[ContextDecisionMaking-v0](#contextdecisionmaking)

[DawTwoStep-v0](#dawtwostep)

[DelayPairedAssociation-v0](#delaypairedassociation)

[DelayComparison-v0](#delayedcomparison)

[DelayMatchCategory-v0](#delayedmatchcategory)

[DelayMatchSample-v0](#delayedmatchtosample)

[DelayMatchSampleDistractor1D-v0](#delayedmatchtosampledistractor1d)

[Detection-v0](#detection)

[EconomicDecisionMaking-v0](#economicdecisionmaking)

[GoNogo-v0](#gonogo)

[IntervalDiscrimination-v0](#intervaldiscrimination)

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

___

Tags: [confidence](#confidence), [context dependent](#context-dependent), [continuous action space](#continuous-action-space), [delayed response](#delayed-response), [go-no-go](#go-no-go), [motor](#motor), [multidimensional action space](#multidimensional-action-space), [n-alternative](#n-alternative), [perceptual](#perceptual), [reaction time](#reaction-time), [steps action space](#steps-action-space), [supervised](#supervised), [timing](#timing), [two-alternative](#two-alternative), [value-based](#value-based), [working memory](#working-memory)

___

### AngleReproduction  
Doc:   
        The agent has to reproduce to two angles separated by a constant delay.  
        dt: Timestep duration. (def: 100 (ms), int)  
        rewards:  
            R_CORRECT: given when correct. (def: +1., float)  
            R_FAIL: given when incorrect. (def: -0.1, float)  
        timing: Description and duration of periods forming a trial.  
          
Reference paper   
[Visual perception as retrospective Bayesian decoding from high- to low-level features](https://www.pnas.org/content/114/43/E9115.short)  
  
Period timing (ms)   
fixation : constant 500  
stim1 : constant 500  
delay1 : constant 500  
stim2 : constant 500  
delay2 : constant 500  
go1 : constant 500  
go2 : constant 500  
  
Reward structure   
correct : 1.0  
fail : -0.1  
  
Tags: [perceptual](#perceptual), [working memory](#working-memory), [delayed response](#delayed-response), [steps action space](#steps-action-space)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/anglereproduction.py)

___

### AntiReach1D  
Doc:   
        The agent has to move in the direction opposite to the one indicated  
        by the observation.  
        dt: Timestep duration. (def: 100 (ms), int)  
        rewards:  
            R_CORRECT: given when correct. (def: +1., float)  
            R_FAIL: given when incorrect. (def: -0.1, float)  
        timing: Description and duration of periods forming a trial.  
          
Reference paper   
[Look away: the anti-saccade task and the voluntary control of eye movement](https://www.nature.com/articles/nrn1345)  
  
Period timing (ms)   
fixation : constant 500  
reach : constant 500  
  
Reward structure   
correct : 1.0  
fail : -0.1  
  
Tags: [perceptual](#perceptual), [steps action space](#steps-action-space)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/antireach.py)

___

### Bandit  
Doc:   
        The agent has to select between N actions with different reward  
        probabilities.  
        dt: Timestep duration. (def: 100 (ms), int)  
        n_arms: Number of arms. (def: 2, int)  
        probs: Reward probabilities for each arm. (def: (.9, .1), tuple)  
        gt_arm: High reward arm. (def: 0, int)  
        rewards:  
            R_CORRECT: given when correct. (def: +1., float)  
        timing: Description and duration of periods forming a trial.  
          
Reference paper   
[Prefrontal cortex as a meta-reinforcement learning system](https://www.nature.com/articles/s41593-018-0147-8)  
  
Reward structure   
correct : 1.0  
  
Tags: [n-alternative](#n-alternative), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/bandit.py)

___

### CVLearning  
Doc:   
        Implements shaping for the delay-response task, in which agents  
        have to integrate two stimuli and report which one is larger on  
        average after a delay.  
        dt: Timestep duration. (def: 100 (ms), int)  
        rewards:  
            R_ABORTED: given when breaking fixation. (def: -0.1, float)  
            R_CORRECT: given when correct. (def: +1., float)  
            R_FAIL: given when incorrect. (def: -1., float)  
        timing: Description and duration of periods forming a trial.  
        stim_scale: Controls the difficulty of the experiment. (def: 1., float)  
        perf_w: Window used to compute the mean reward. (def: 1000, int)  
        max_num_reps: Maximum number of times that agent can go in a row  
        to the same side during phase 0. (def: 3, int)  
        init_ph: Phase initializing the task. (def: 0, int)  
        th: Performance threshold needed to proceed to the following phase.  
        (def: 0.8, float)  
          
Reference paper   
[Discrete attractor dynamics underlies persistent activity in the frontal cortex](https://www.nature.com/articles/s41586-019-0919-7)  
  
Period timing (ms)   
fixation : constant 200  
stimulus : constant 1150  
delay : choice [300, 500, 700, 900, 1200, 2000, 3200, 4000]  
decision : constant 1500  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : -1.0  
  
Tags: [perceptual](#perceptual), [delayed response](#delayed-response), [two-alternative](#two-alternative), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/cv_learning.py)

___

### ChangingEnvironment  
Doc:   
        Random Dots Motion tasks in which the correct action  
        depends on a randomly changing context.  
        dt: Timestep duration. (def: 100 (ms), int)  
        rewards:  
            R_ABORTED: given when breaking fixation. (def: -0.1, float)  
            R_CORRECT: given when correct. (def: +1., float)  
            R_FAIL: given when incorrect. (def: 0., float)  
        timing: Description and duration of periods forming a trial.  
        stim_scale: Controls the difficulty of the experiment. (def: 1., float)  
        cxt_ch_prob: Probability of changing context. (def: 0.01, float)  
        cxt_cue: Whether to show context as a cue. (def: False, bool)  
          
Reference paper   
[Hierarchical decision processes that operate over distinct timescales underlie choice and changes in strategy](https://www.pnas.org/content/113/31/E4531)  
  
Period timing (ms)   
fixation : constant 500  
stimulus : truncated_exponential [1000, 500, 1500]  
decision : constant 500  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : 0.0  
  
Tags: [perceptual](#perceptual), [two-alternative](#two-alternative), [supervised](#supervised), [context dependent](#context-dependent)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/changingenvironment.py)

___

### ContextDecisionMaking  
Doc:   
        Agent has to perform one of two different perceptual discriminations.  
        On every trial, a contextual cue indicates which one to perform.  
        dt: Timestep duration. (def: 100 (ms), int)  
        rewards:  
            R_ABORTED: given when breaking fixation. (def: -0.1, float)  
            R_CORRECT: given when correct. (def: +1., float)  
        timing: Description and duration of periods forming a trial.  
          
Reference paper   
[Context-dependent computation by recurrent dynamics in prefrontal cortex](https://www.nature.com/articles/nature12742)  
  
Period timing (ms)   
fixation : constant 300  
stimulus : constant 750  
delay : truncated_exponential [600, 300, 3000]  
decision : constant 100  
  
Reward structure   
abort : -0.1  
correct : 1.0  
  
Tags: [perceptual](#perceptual), [context dependent](#context-dependent), [two-alternative](#two-alternative), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/contextdecisionmaking.py)

___

### DawTwoStep  
Doc:   
        On each trial, an initial choice between two options lead  
        to either of two, second-stage states. In turn, these both  
        demand another two-option choice, each of which is associated  
        with a different chance of receiving reward.  
        dt: Timestep duration. (def: 100 (ms), int)  
        rewards:  
            R_ABORTED: given when breaking fixation. (def: -0.1, float)  
            R_CORRECT: given when correct. (def: +1., float)  
        timing: Description and duration of periods forming a trial.  
          
Reference paper   
[Model-Based Influences on Humans Choices and Striatal Prediction Errors](https://www.sciencedirect.com/science/article/pii/S0896627311001255)  
  
Reward structure   
abort : -0.1  
correct : 1.0  
  
Tags: [two-alternative](#two-alternative)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/dawtwostep.py)

___

### DelayPairedAssociation  
Doc:   
        A sample is followed by a delay and a test. Agents have to report if  
        the pair sample-test is a rewarded pair or not.  
        dt: Timestep duration. (def: 100 (ms), int)  
        rewards: dictionary of rewards  
        timing: Description and duration of periods forming a trial.  
        noise: Standard deviation of the Gaussian noise added to  
        the stimulus. (def: 0.01, float)  
          
Reference paper   
[Active information maintenance in working memory by a sensory cortex](https://elifesciences.org/articles/43191)  
  
Period timing (ms)   
fixation : constant 0  
stim1 : constant 1000  
delay_btw_stim : constant 13000  
stim2 : constant 1000  
delay_aft_stim : constant 1000  
decision : constant 500  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : -1.0  
miss : 0.0  
  
Tags: [perceptual](#perceptual), [working memory](#working-memory), [go-no-go](#go-no-go), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/delaypairedassociation.py)

___

### DelayComparison  
Doc:   
        Two-alternative forced choice task in which the subject  
        has to compare two stimuli separated by a delay to decide  
        which one has a higher frequency.  
        dt: Timestep duration. (def: 100 (ms), int)  
        rewards:  
            R_ABORTED: given when breaking fixation. (def: -0.1, float)  
            R_CORRECT: given when correct. (def: +1., float)  
            R_FAIL: given when incorrect. (def: 0., float)  
        timing: Description and duration of periods forming a trial.  
          
Reference paper   
[Neuronal Population Coding of Parametric Working Memory](https://www.jneurosci.org/content/30/28/9424)  
  
Period timing (ms)   
fixation : uniform (1500, 3000)  
f1 : constant 500  
delay : constant 3000  
f2 : constant 500  
decision : constant 100  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : 0.0  
  
Tags: [perceptual](#perceptual), [working memory](#working-memory), [two-alternative](#two-alternative), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/delayedcomparison.py)

___

### DelayMatchCategory  
Doc:   
        A sample stimulus is followed by a delay and test. Agents are required  
        to indicate if the sample and test are in the same category.  
        dt: Timestep duration.  
        rewards:  
            R_ABORTED: given when breaking fixation. (def: -0.1, float)  
            R_CORRECT: given when correct. (def: +1., float)  
            R_FAIL: given when incorrect. (def: 0., float)  
        timing: Description and duration of periods forming a trial.  
          
Reference paper   
[Experience-dependent representation of visual categories in parietal cortex](https://www.nature.com/articles/nature05078)  
  
Period timing (ms)   
fixation : constant 500  
sample : constant 650  
first_delay : constant 1000  
test : constant 650  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : 0.0  
  
Tags: [perceptual](#perceptual), [working memory](#working-memory), [two-alternative](#two-alternative), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/delaymatchcategory.py)

___

### DelayMatchSample  
Doc:   
        A sample stimulus is followed by a delay and test. Agents are required  
        to indicate if the sample and test are the same stimulus.  
        dt: Timestep duration. (def: 100 (ms), int)  
        rewards:  
            R_ABORTED: given when breaking fixation. (def: -0.1, float)  
            R_CORRECT: given when correct. (def: +1., float)  
            R_FAIL: given when incorrect. (def: 0., float)  
        timing: Description and duration of periods forming a trial.  
          
Reference paper   
[Neural Mechanisms of Visual Working Memory in Prefrontal Cortex of the Macaque](https://www.jneurosci.org/content/jneuro/16/16/5154.full.pdf)  
  
Period timing (ms)   
fixation : constant 300  
sample : constant 500  
delay : constant 1000  
test : constant 500  
decision : constant 900  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : 0.0  
  
Tags: [perceptual](#perceptual), [working memory](#working-memory), [two-alternative](#two-alternative), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/delaymatchsample.py)

___

### DelayMatchSampleDistractor1D  
Doc:   
        Delay Match to sample with multiple, potentially repeating distractors.  
        dt: Timestep duration. (def: 100 (ms), int)  
        rewards:  
            R_ABORTED: given when breaking fixation. (def: -0.1, float)  
            R_CORRECT: given when correct. (def: +1., float)  
            R_FAIL: given when incorrect. (def: -1., float)  
        timing: Description and duration of periods forming a trial.  
          
Reference paper   
[Neural Mechanisms of Visual Working Memory in Prefrontal Cortex of the Macaque](https://www.jneurosci.org/content/jneuro/16/16/5154.full.pdf)  
  
Period timing (ms)   
fixation : constant 300  
sample : constant 500  
delay1 : constant 1000  
test1 : constant 500  
delay2 : constant 1000  
test2 : constant 500  
delay3 : constant 1000  
test3 : constant 500  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : -1.0  
  
Tags: [perceptual](#perceptual), [working memory](#working-memory), [two-alternative](#two-alternative), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/delaymatchsample.py)

___

### Detection  
Doc:   
        The agent has to GO if a stimulus is presented.  
        dt: Timestep duration. (def: 100 (ms), int)  
        rewards: dictionary of rewards  
        timing: Description and duration of periods forming a trial.  
        noise: Standard deviation of background noise. (def: 1., float)  
        delay: If not None indicates the delay, from the moment of the start of  
        the stimulus period when the actual stimulus is presented. Otherwise,  
        the delay is drawn from a uniform distribution. (def: None (ms), int)  
        stim_dur: Stimulus duration. (def: 100 (ms), int)  
          
Reference paper   
Missing paper name  
Missing paper link  
  
Period timing (ms)   
fixation : constant 500  
stimulus : truncated_exponential [1000, 500, 1500]  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : -1.0  
miss : -1  
  
Tags: [perceptual](#perceptual), [reaction time](#reaction-time), [go-no-go](#go-no-go), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/detection.py)

___

### EconomicDecisionMaking  
Doc:   
        Agents choose between two stimuli (A and B; where A is preferred)  
        offered in different amounts.  
        dt: Timestep duration. (def: 100 (ms), int)  
        rewards: dictionary of rewards  
        timing: Description and duration of periods forming a trial.  
          
Reference paper   
[Neurons in the orbitofrontal cortex encode economic value](https://www.nature.com/articles/nature04676)  
  
Period timing (ms)   
fixation : constant 1500  
offer_on : uniform [1000, 2000]  
decision : constant 750  
  
Reward structure   
abort : -0.1  
correct : 0.22  
  
Tags: [perceptual](#perceptual), [value-based](#value-based)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/economicdecisionmaking.py)

___

### GoNogo  
Doc:   
        Go/No-Go task in which the subject has either Go (e.g. lick)  
        or not Go depending on which one of two stimuli is presented with.  
        dt: Timestep duration. (def: 100 (ms), int)  
        rewards: reward dictionary  
        timing: Description and duration of periods forming a trial.  
          
Reference paper   
[Active information maintenance in working memory by a sensory cortex](https://elifesciences.org/articles/43191)  
  
Period timing (ms)   
fixation : constant 0  
stimulus : constant 500  
resp_delay : constant 500  
decision : constant 500  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : -0.5  
miss : -0.5  
  
Tags: [delayed response](#delayed-response), [go-no-go](#go-no-go), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/gonogo.py)

___

### IntervalDiscrimination  
Doc:   
        Agents have to report which of two stimuli presented  
        sequentially is longer.  
        dt: Timestep duration. (def: 80 (ms), int)  
        rewards: dictionary of rewards  
        timing: Description and duration of periods forming a trial.  
          
Reference paper   
[Feature- and Order-Based Timing Representations in the Frontal Cortex](https://www.sciencedirect.com/science/article/pii/S0896627309004887)  
  
Period timing (ms)   
fixation : constant 300  
stim1 : uniform (300, 600)  
delay1 : choice [800, 1500]  
stim2 : uniform (300, 600)  
delay2 : constant 500  
decision : constant 300  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : 0.0  
  
Tags: [timing](#timing), [working memory](#working-memory), [delayed response](#delayed-response), [two-alternative](#two-alternative), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/intervaldiscrimination.py)

___

### MatchingPenny  
Doc:   
        The agent is rewarded when it selects the same target as the computer.  
        dt: Timestep duration. (def: 100 (ms), int)  
        opponent_type: Type of opponent. (def: 'mean_action', str)  
        rewards:  
            R_CORRECT: given when correct. (def: +1., float)  
            R_FAIL: given when incorrect. (def: 0., float)  
        timing: Description and duration of periods forming a trial.  
        learning_rate: learning rate in the mean_action opponent  
          
Reference paper   
[Prefrontal cortex and decision making in a mixed-strategy game](https://www.nature.com/articles/nn1209)  
  
Reward structure   
correct : 1.0  
fail : 0.0  
  
Tags: [two-alternative](#two-alternative), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/matchingpenny.py)

___

### MotorTiming  
Doc:   
        Agents have to produce different time intervals  
        using different effectors (actions).  
        dt: Timestep duration. (def: 80 (ms), int)  
        rewards:  
            R_ABORTED: given when breaking fixation. (def: -0.1, float)  
            R_CORRECT: given when correct. (def: +1., float)  
            R_FAIL: given when incorrect. (def: 0., float)  
        timing: Description and duration of periods forming a trial.  
        prod_margin: controls the interval around the ground truth production  
                    time within which the agent receives proportional reward  
          
Reference paper   
[Flexible timing by temporal scaling of cortical responses](https://www.nature.com/articles/s41593-017-0028-6)  
  
Period timing (ms)   
fixation : constant 500  
cue : uniform [1000, 3000]  
set : constant 50  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : 0.0  
  
Tags: [timing](#timing), [go-no-go](#go-no-go), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/readysetgo.py)

___

### nalt_PerceptualDecisionMaking  
Doc:   
        N-alternative forced choice task in which the subject has  
        to integrate N stimuli to decide which one is higher on average.  
        dt: Timestep duration. (def: 100 (ms), int)  
        rewards:  
            R_ABORTED: given when breaking fixation. (def: -0.1, float)  
            R_CORRECT: given when correct. (def: +1., float)  
            R_FAIL: given when incorrect. (def: 0., float)  
        timing: Description and duration of periods forming a trial.  
        stim_scale: Controls the difficulty of the experiment. (def: 1., float)  
        n_ch: Number of choices. (def: 3, int)  
          
Reference paper   
Missing paper name  
Missing paper link  
  
Period timing (ms)   
fixation : constant 500  
stimulus : truncated_exponential [330, 80, 1500]  
decision : constant 500  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : 0.0  
  
Tags: [perceptual](#perceptual), [n-alternative](#n-alternative), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/nalt_perceptualdecisionmaking.py)

___

### PerceptualDecisionMaking  
Doc:   
        Two-alternative forced choice task in which the subject has to  
        integrate two stimuli to decide which one is higher on average.  
  
        Parameters:  
        dt: Timestep duration. (def: 100 (ms), int)  
        rewards:  
            R_ABORTED: given when breaking fixation. (def: -0.1, float)  
            R_CORRECT: given when correct. (def: +1., float)  
            R_FAIL: given when incorrect. (def: 0., float)  
        timing: Description and duration of periods forming a trial.  
        stim_scale: Controls the difficulty of the experiment. (def: 1., float)  
          
Reference paper   
[The analysis of visual motion: a comparison of neuronal and psychophysical performance](https://www.jneurosci.org/content/12/12/4745)  
  
Period timing (ms)   
fixation : constant 100  
stimulus : constant 2000  
decision : constant 100  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : 0.0  
  
Tags: [perceptual](#perceptual), [two-alternative](#two-alternative), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/perceptualdecisionmaking.py)

___

### PerceptualDecisionMakingDelayResponse  
Doc:   
        Agents have to integrate two stimuli and report which one is  
        larger on average after a delay.  
        dt: Timestep duration. (def: 100 (ms), int)  
        rewards:  
            R_ABORTED: given when breaking fixation. (def: -0.1, float)  
            R_CORRECT: given when correct. (def: +1., float)  
            R_FAIL: given when incorrect. (def: -1., float)  
        timing: Description and duration of periods forming a trial.  
        stim_scale: Controls the difficulty of the experiment. (def: 1., float)  
          
Reference paper   
[Discrete attractor dynamics underlies persistent activity in the frontal cortex](https://www.nature.com/articles/s41586-019-0919-7)  
  
Period timing (ms)   
fixation : constant 0  
stimulus : constant 1150  
delay : choice [300, 500, 700, 900, 1200, 2000, 3200, 4000]  
decision : constant 1500  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : 0.0  
  
Other parameters:   
stim_scale : Controls the difficulty of the experiment. (def: 1.)  
  
Tags: [perceptual](#perceptual), [delayed response](#delayed-response), [two-alternative](#two-alternative), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/perceptualdecisionmaking.py)

___

### PostDecisionWager  
Doc:   
        Agents do a discrimination task(see PerceptualDecisionMaking). On a  
        random half of the trials, the agent is given the option to abort  
        the direction discrimination and to choose instead a small but  
        certain reward associated with a action.  
        dt: Timestep duration. (def: 100 (ms), int)  
        rewards:  
            R_ABORTED: given when breaking fixation. (def: -0.1, float)  
            R_CORRECT: given when correct. (def: +1., float)  
            R_FAIL: given when incorrect. (def: 0., float)  
        timing: Description and duration of periods forming a trial.  
          
Reference paper   
[Representation of Confidence Associated with a Decision by Neurons in the Parietal Cortex](https://science.sciencemag.org/content/324/5928/759.long)  
  
Period timing (ms)   
fixation : constant 100  
stimulus : truncated_exponential [180, 100, 900]  
delay : truncated_exponential [1350, 1200, 1800]  
pre_sure : uniform [500, 750]  
decision : constant 100  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : 0.0  
sure : 0.7  
  
Tags: [perceptual](#perceptual), [delayed response](#delayed-response), [confidence](#confidence)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/postdecisionwager.py)

___

### Reaching1D  
Doc:   
        The agent has to reproduce the angle indicated by the observation.  
        dt: Timestep duration. (def: 100 (ms), int)  
        rewards:  
            R_CORRECT: given when correct. (def: +1., float)  
            R_FAIL: given when incorrect. (def: -0.1, float)  
        timing: Description and duration of periods forming a trial.  
          
Reference paper   
[Neuronal population coding of movement direction](https://science.sciencemag.org/content/233/4771/1416)  
  
Period timing (ms)   
fixation : constant 500  
reach : constant 500  
  
Reward structure   
correct : 1.0  
fail : -0.1  
  
Tags: [motor](#motor), [steps action space](#steps-action-space)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/reaching.py)

___

### Reaching1DWithSelfDistraction  
Doc:   
        The agent has to reproduce the angle indicated by the observation.  
        Furthermore, the reaching state itself generates strong inputs that  
        overshadows the actual target input.  
        dt: Timestep duration. (def: 100 (ms), int)  
        rewards:  
            R_CORRECT: given when correct. (def: +1., float)  
            R_FAIL: given when incorrect. (def: -0.1, float)  
        timing: Description and duration of periods forming a trial.  
          
Reference paper   
Missing paper name  
Missing paper link  
  
Period timing (ms)   
fixation : constant 500  
reach : constant 500  
  
Reward structure   
correct : 1.0  
fail : -0.1  
  
Tags: [motor](#motor), [steps action space](#steps-action-space)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/reaching.py)

___

### ReachingDelayResponse  
Doc:   
        Working memory visual spatial task ~ Funahashi et al. 1991 adapted to  
        freely moving mice in a continous choice-space.  
        dt: Timestep duration. (def: 100 (ms), int)  
        rewards: dictionary of rewards  
        timing: Description and duration of periods forming a trial.  
          
Reference paper   
Missing paper name  
Missing paper link  
  
Period timing (ms)   
stimulus : constant 500  
delay : choice [0, 1000, 2000]  
decision : constant 5000  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : -0.0  
miss : -0.5  
  
Tags: [perceptual](#perceptual), [delayed response](#delayed-response), [continuous action space](#continuous-action-space), [multidimensional action space](#multidimensional-action-space), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/reachingdelayresponse.py)

___

### ReadySetGo  
Doc:   
        Agents have to measure and produce different time intervals.  
        dt: Timestep duration. (def: 80 (ms), int)  
        rewards: dictionary of rewards  
        timing: Description and duration of periods forming a trial.  
        gain: Controls the measure that the agent has to produce. (def: 1, int)  
        prod_margin: controls the interval around the ground truth production  
                    time within which the agent receives proportional reward  
          
Reference paper   
[Flexible Sensorimotor Computations through Rapid Reconfiguration of Cortical Dynamics](https://www.sciencedirect.com/science/article/pii/S0896627318304185)  
  
Period timing (ms)   
fixation : constant 100  
ready : constant 83  
measure : choice [800, 1500]  
set : constant 83  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : 0.0  
  
Tags: [timing](#timing), [go-no-go](#go-no-go), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/readysetgo.py)

___

### Tags ### 

### confidence 

[PostDecisionWager-v0](#postdecisionwager)

### context dependent 

[ChangingEnvironment-v0](#changingenvironment)

[ContextDecisionMaking-v0](#contextdecisionmaking)

### continuous action space 

[ReachingDelayResponse-v0](#reachingdelayresponse)

### delayed response 

[AngleReproduction-v0](#anglereproduction)

[CVLearning-v0](#cvlearning)

[GoNogo-v0](#gonogo)

[IntervalDiscrimination-v0](#intervaldiscrimination)

[PerceptualDecisionMakingDelayResponse-v0](#perceptualdecisionmakingdelayresponse)

[PostDecisionWager-v0](#postdecisionwager)

[ReachingDelayResponse-v0](#reachingdelayresponse)

### go-no-go 

[DelayPairedAssociation-v0](#delaypairedassociation)

[Detection-v0](#detection)

[GoNogo-v0](#gonogo)

[MotorTiming-v0](#motortiming)

[ReadySetGo-v0](#readysetgo)

### motor 

[Reaching1D-v0](#reaching1d)

[Reaching1DWithSelfDistraction-v0](#reaching1dwithselfdistraction)

### multidimensional action space 

[ReachingDelayResponse-v0](#reachingdelayresponse)

### n-alternative 

[Bandit-v0](#bandit)

[NAltPerceptualDecisionMaking-v0](#nalt_perceptualdecisionmaking)

### perceptual 

[AngleReproduction-v0](#anglereproduction)

[AntiReach-v0](#antireach1d)

[CVLearning-v0](#cvlearning)

[ChangingEnvironment-v0](#changingenvironment)

[ContextDecisionMaking-v0](#contextdecisionmaking)

[DelayPairedAssociation-v0](#delaypairedassociation)

[DelayComparison-v0](#delayedcomparison)

[DelayMatchCategory-v0](#delayedmatchcategory)

[DelayMatchSample-v0](#delayedmatchtosample)

[DelayMatchSampleDistractor1D-v0](#delayedmatchtosampledistractor1d)

[Detection-v0](#detection)

[EconomicDecisionMaking-v0](#economicdecisionmaking)

[NAltPerceptualDecisionMaking-v0](#nalt_perceptualdecisionmaking)

[PerceptualDecisionMaking-v0](#perceptualdecisionmaking)

[PerceptualDecisionMakingDelayResponse-v0](#perceptualdecisionmakingdelayresponse)

[PostDecisionWager-v0](#postdecisionwager)

[ReachingDelayResponse-v0](#reachingdelayresponse)

### reaction time 

[Detection-v0](#detection)

### steps action space 

[AngleReproduction-v0](#anglereproduction)

[AntiReach-v0](#antireach1d)

[Reaching1D-v0](#reaching1d)

[Reaching1DWithSelfDistraction-v0](#reaching1dwithselfdistraction)

### supervised 

[Bandit-v0](#bandit)

[CVLearning-v0](#cvlearning)

[ChangingEnvironment-v0](#changingenvironment)

[ContextDecisionMaking-v0](#contextdecisionmaking)

[DelayPairedAssociation-v0](#delaypairedassociation)

[DelayComparison-v0](#delayedcomparison)

[DelayMatchCategory-v0](#delayedmatchcategory)

[DelayMatchSample-v0](#delayedmatchtosample)

[DelayMatchSampleDistractor1D-v0](#delayedmatchtosampledistractor1d)

[Detection-v0](#detection)

[GoNogo-v0](#gonogo)

[IntervalDiscrimination-v0](#intervaldiscrimination)

[MatchingPenny-v0](#matchingpenny)

[MotorTiming-v0](#motortiming)

[NAltPerceptualDecisionMaking-v0](#nalt_perceptualdecisionmaking)

[PerceptualDecisionMaking-v0](#perceptualdecisionmaking)

[PerceptualDecisionMakingDelayResponse-v0](#perceptualdecisionmakingdelayresponse)

[ReachingDelayResponse-v0](#reachingdelayresponse)

[ReadySetGo-v0](#readysetgo)

### timing 

[IntervalDiscrimination-v0](#intervaldiscrimination)

[MotorTiming-v0](#motortiming)

[ReadySetGo-v0](#readysetgo)

### two-alternative 

[CVLearning-v0](#cvlearning)

[ChangingEnvironment-v0](#changingenvironment)

[ContextDecisionMaking-v0](#contextdecisionmaking)

[DawTwoStep-v0](#dawtwostep)

[DelayComparison-v0](#delayedcomparison)

[DelayMatchCategory-v0](#delayedmatchcategory)

[DelayMatchSample-v0](#delayedmatchtosample)

[DelayMatchSampleDistractor1D-v0](#delayedmatchtosampledistractor1d)

[IntervalDiscrimination-v0](#intervaldiscrimination)

[MatchingPenny-v0](#matchingpenny)

[PerceptualDecisionMaking-v0](#perceptualdecisionmaking)

[PerceptualDecisionMakingDelayResponse-v0](#perceptualdecisionmakingdelayresponse)

### value-based 

[EconomicDecisionMaking-v0](#economicdecisionmaking)

### working memory 

[AngleReproduction-v0](#anglereproduction)

[DelayPairedAssociation-v0](#delaypairedassociation)

[DelayComparison-v0](#delayedcomparison)

[DelayMatchCategory-v0](#delayedmatchcategory)

[DelayMatchSample-v0](#delayedmatchtosample)

[DelayMatchSampleDistractor1D-v0](#delayedmatchtosampledistractor1d)

[IntervalDiscrimination-v0](#intervaldiscrimination)

