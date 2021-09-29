* Under development, details subject to change

### List of 29 tasks implemented

[AntiReach-v0](#antireach)

[Bandit-v0](#bandit)

[ContextDecisionMaking-v0](#contextdecisionmaking)

[DawTwoStep-v0](#dawtwostep)

[DelayComparison-v0](#delaycomparison)

[DelayMatchCategory-v0](#delaymatchcategory)

[DelayMatchSample-v0](#delaymatchsample)

[DelayMatchSampleDistractor1D-v0](#delaymatchsampledistractor1d)

[DelayPairedAssociation-v0](#delaypairedassociation)

[DualDelayMatchSample-v0](#dualdelaymatchsample)

[EconomicDecisionMaking-v0](#economicdecisionmaking)

[GoNogo-v0](#gonogo)

[HierarchicalReasoning-v0](#hierarchicalreasoning)

[IntervalDiscrimination-v0](#intervaldiscrimination)

[MotorTiming-v0](#motortiming)

[MultiSensoryIntegration-v0](#multisensoryintegration)

[Null-v0](#null)

[OneTwoThreeGo-v0](#onetwothreego)

[PerceptualDecisionMaking-v0](#perceptualdecisionmaking)

[PerceptualDecisionMakingDelayResponse-v0](#perceptualdecisionmakingdelayresponse)

[PostDecisionWager-v0](#postdecisionwager)

[ProbabilisticReasoning-v0](#probabilisticreasoning)

[PulseDecisionMaking-v0](#pulsedecisionmaking)

[Reaching1D-v0](#reaching1d)

[Reaching1DWithSelfDistraction-v0](#reaching1dwithselfdistraction)

[ReachingDelayResponse-v0](#reachingdelayresponse)

[ReadySetGo-v0](#readysetgo)

[SingleContextDecisionMaking-v0](#singlecontextdecisionmaking)

[SpatialSuppressMotion-v0](#spatialsuppressmotion)

___

Tags: [confidence](#confidence), [context dependent](#context-dependent), [continuous action space](#continuous-action-space), [delayed response](#delayed-response), [go-no-go](#go-no-go), [motor](#motor), [multidimensional action space](#multidimensional-action-space), [n-alternative](#n-alternative), [perceptual](#perceptual), [reaction time](#reaction-time), [steps action space](#steps-action-space), [supervised](#supervised), [timing](#timing), [two-alternative](#two-alternative), [value-based](#value-based), [working memory](#working-memory)

___

### AntiReach  
Doc: Anti-response task.  
  
    During the fixation period, the agent fixates on a fixation point.  
    During the following stimulus period, the agent is then shown a stimulus away  
    from the fixation point. Finally, the agent needs to respond in the  
    opposite direction of the stimulus during the decision period.  
  
    Args:  
        anti: bool, if True, requires an anti-response. If False, requires a  
            pro-response, i.e. response towards the stimulus.  
      
Reference paper   
[Look away: the anti-saccade task and the voluntary control of eye movement](https://www.nature.com/articles/nrn1345)  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : 0.0  
  
Tags: [perceptual](#perceptual), [steps action space](#steps-action-space)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/antireach.py)

___

### Bandit  
Doc: Multi-arm bandit task.  
  
    On each trial, the agent is presented with multiple choices. Each  
    option produces a reward of a certain magnitude given a certain probability.  
  
    Args:  
        n: int, the number of choices (arms)  
        p: tuple of length n, describes the probability of each arm  
            leading to reward  
        rewards: tuple of length n, describe the reward magnitude of each option when rewarded  
      
Reference paper   
[Prefrontal cortex as a meta-reinforcement learning system](https://www.nature.com/articles/s41593-018-0147-8)  
  
Reward structure   
[1. 1.]  
Tags: [n-alternative](#n-alternative)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/bandit.py)

___

### ContextDecisionMaking  
Doc: Context-dependent decision-making task.  
  
    The agent simultaneously receives stimulus inputs from two modalities (  
    for example, a colored random dot motion pattern with color and motion  
    modalities). The agent needs to make a perceptual decision based on  
    only one of the two modalities, while ignoring the other. The relevant  
    modality is explicitly indicated by a rule signal.  
      
Reference paper   
[Context-dependent computation by recurrent dynamics in prefrontal cortex](https://www.nature.com/articles/nature12742)  
  
Reward structure   
abort : -0.1  
correct : 1.0  
  
Tags: [perceptual](#perceptual), [context dependent](#context-dependent), [two-alternative](#two-alternative), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/contextdecisionmaking.py)

___

### DawTwoStep  
Doc: Daw Two-step task.  
  
    On each trial, an initial choice between two options lead  
    to either of two, second-stage states. In turn, these both  
    demand another two-option choice, each of which is associated  
    with a different chance of receiving reward.  
      
Reference paper   
[Model-Based Influences on Humans Choices and Striatal Prediction Errors](https://www.sciencedirect.com/science/article/pii/S0896627311001255)  
  
Reward structure   
abort : -0.1  
correct : 1.0  
  
Tags: [two-alternative](#two-alternative)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/dawtwostep.py)

___

### DelayComparison  
Doc: Delayed comparison.  
  
    The agent needs to compare the magnitude of two stimuli are separated by a  
    delay period. The agent reports its decision of the stronger stimulus  
    during the decision period.  
      
Reference paper   
[Neuronal Population Coding of Parametric Working Memory](https://www.jneurosci.org/content/30/28/9424)  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : 0.0  
  
Tags: [perceptual](#perceptual), [working memory](#working-memory), [two-alternative](#two-alternative), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/delaycomparison.py)

___

### DelayMatchCategory  
Doc: Delayed match-to-category task.  
  
    A sample stimulus is shown during the sample period. The stimulus is  
    characterized by a one-dimensional variable, such as its orientation  
    between 0 and 360 degree. This one-dimensional variable is separated  
    into two categories (for example, 0-180 degree and 180-360 degree).  
    After a delay period, a test stimulus is shown. The agent needs to  
    determine whether the sample and the test stimuli belong to the same  
    category, and report that decision during the decision period.  
      
Reference paper   
[Experience-dependent representation of visual categories in parietal cortex](https://www.nature.com/articles/nature05078)  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : 0.0  
  
Tags: [perceptual](#perceptual), [working memory](#working-memory), [two-alternative](#two-alternative), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/delaymatchcategory.py)

___

### DelayMatchSample  
Doc: Delayed match-to-sample task.  
  
    A sample stimulus is shown during the sample period. The stimulus is  
    characterized by a one-dimensional variable, such as its orientation  
    between 0 and 360 degree. After a delay period, a test stimulus is  
    shown. The agent needs to determine whether the sample and the test  
    stimuli are equal, and report that decision during the decision period.  
      
Reference paper   
[Neural Mechanisms of Visual Working Memory in Prefrontal Cortex of the Macaque](https://www.jneurosci.org/content/jneuro/16/16/5154.full.pdf)  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : 0.0  
  
Tags: [perceptual](#perceptual), [working memory](#working-memory), [two-alternative](#two-alternative), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/delaymatchsample.py)

___

### DelayMatchSampleDistractor1D  
Doc: Delayed match-to-sample with multiple, potentially repeating  
    distractors.  
  
    A sample stimulus is shown during the sample period. The stimulus is  
    characterized by a one-dimensional variable, such as its orientation  
    between 0 and 360 degree. After a delay period, the first test stimulus is  
    shown. The agent needs to determine whether the sample and this test  
    stimuli are equal. If so, it needs to produce the match response. If the  
    first test is not equal to the sample stimulus, another delay period and  
    then a second test stimulus follow, and so on.  
      
Reference paper   
[Neural Mechanisms of Visual Working Memory in Prefrontal Cortex of the Macaque](https://www.jneurosci.org/content/jneuro/16/16/5154.full.pdf)  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : -1.0  
  
Tags: [perceptual](#perceptual), [working memory](#working-memory), [two-alternative](#two-alternative), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/delaymatchsample.py)

___

### DelayPairedAssociation  
Doc: Delayed paired-association task.  
  
    The agent is shown a pair of two stimuli separated by a delay period. For  
    half of the stimuli-pairs shown, the agent should choose the Go response.  
    The agent is rewarded if it chose the Go response correctly.  
      
Reference paper   
[Active information maintenance in working memory by a sensory cortex](https://elifesciences.org/articles/43191)  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : -1.0  
miss : 0.0  
  
Tags: [perceptual](#perceptual), [working memory](#working-memory), [go-no-go](#go-no-go), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/delaypairedassociation.py)

___

### DualDelayMatchSample  
Doc: Two-item Delay-match-to-sample.  
  
    The trial starts with a fixation period. Then during the sample period,  
    two sample stimuli are shown simultaneously. Followed by the first delay  
    period, a cue is shown, indicating which sample stimulus will be tested.  
    Then the first test stimulus is shown and the agent needs to report whether  
    this test stimulus matches the cued sample stimulus. Then another delay  
    and then test period follows, and the agent needs to report whether the  
    other sample stimulus matches the second test stimulus.  
      
Reference paper   
[Reactivation of latent working memories with transcranial magnetic stimulation](https://science.sciencemag.org/content/354/6316/1136)  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : 0.0  
  
Tags: [perceptual](#perceptual), [working memory](#working-memory), [two-alternative](#two-alternative), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/dualdelaymatchsample.py)

___

### EconomicDecisionMaking  
Doc: Economic decision making task.  
  
    A agent chooses between two options. Each option offers a certain amount of  
    juice. Its amount is indicated by the stimulus. The two options offer  
    different types of juice, and the agent prefers one over another.  
      
Reference paper   
[Neurons in the orbitofrontal cortex encode economic value](https://www.nature.com/articles/nature04676)  
  
Reward structure   
abort : -0.1  
correct : 0.22  
  
Tags: [perceptual](#perceptual), [value-based](#value-based)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/economicdecisionmaking.py)

___

### GoNogo  
Doc: Go/No-go task.  
  
    A stimulus is shown during the stimulus period. The stimulus period is  
    followed by a delay period, and then a decision period. If the stimulus is  
    a Go stimulus, then the subject should choose the action Go during the  
    decision period, otherwise, the subject should remain fixation.  
      
Reference paper   
[Active information maintenance in working memory by a sensory cortex](https://elifesciences.org/articles/43191)  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : -0.5  
miss : -0.5  
  
Tags: [delayed response](#delayed-response), [go-no-go](#go-no-go), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/gonogo.py)

___

### HierarchicalReasoning  
Doc: Hierarchical reasoning of rules.  
  
    On each trial, the subject receives two flashes separated by a delay  
    period. The subject needs to judge whether the duration of this delay  
    period is shorter than a threshold. Both flashes appear at the  
    same location on each trial. For one trial type, the network should  
    report its decision by going to the location of the flashes if the delay is  
    shorter than the threshold. In another trial type, the network should go to  
    the opposite direction of the flashes if the delay is short.  
    The two types of trials are alternated across blocks, and the block  
    transtion is unannouced.  
      
Reference paper   
[Hierarchical reasoning by neural circuits in the frontal cortex](https://science.sciencemag.org/content/364/6441/eaav8911)  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : 0.0  
  
Tags: [perceptual](#perceptual), [two-alternative](#two-alternative), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/hierarchicalreasoning.py)

___

### IntervalDiscrimination  
Doc: Comparing the time length of two stimuli.  
  
    Two stimuli are shown sequentially, separated by a delay period. The  
    duration of each stimulus is randomly sampled on each trial. The  
    subject needs to judge which stimulus has a longer duration, and reports  
    its decision during the decision period by choosing one of the two  
    choice options.  
      
Reference paper   
[Feature- and Order-Based Timing Representations in the Frontal Cortex](https://www.sciencedirect.com/science/article/pii/S0896627309004887)  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : 0.0  
  
Tags: [timing](#timing), [working memory](#working-memory), [delayed response](#delayed-response), [two-alternative](#two-alternative), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/intervaldiscrimination.py)

___

### MotorTiming  
Doc: Agents have to produce different time intervals  
    using different effectors (actions).  
  
    Args:  
        prod_margin: controls the interval around the ground truth production  
                    time within which the agent receives proportional reward  
      
Reference paper   
[Flexible timing by temporal scaling of cortical responses](https://www.nature.com/articles/s41593-017-0028-6)  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : 0.0  
  
Tags: [timing](#timing), [go-no-go](#go-no-go), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/readysetgo.py)

___

### MultiSensoryIntegration  
Doc: Multi-sensory integration.  
  
    Two stimuli are shown in two input modalities. Each stimulus points to  
    one of the possible responses with a certain strength (coherence). The  
    correct choice is the response with the highest summed strength from  
    both stimuli. The agent is therefore encouraged to integrate information  
    from both modalities equally.  
      
Reference paper   
Missing paper name  
Missing paper link  
  
Reward structure   
abort : -0.1  
correct : 1.0  
  
Tags: [perceptual](#perceptual), [two-alternative](#two-alternative), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/multisensory.py)

___

### Null  
Doc: Null task.  
Reference paper   
Missing paper name  
Missing paper link  
  
Reward structure   
  
Other parameters:   
render.modes : []  Tags

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/null.py)

___

### OneTwoThreeGo  
Doc: Agents reproduce time intervals based on two samples.  
  
    Args:  
        prod_margin: controls the interval around the ground truth production  
                    time within which the agent receives proportional reward  
      
Reference paper   
[Internal models of sensorimotor integration regulate cortical dynamics](https://www.nature.com/articles/s41593-019-0500-6)  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : 0.0  
  
Tags: [timing](#timing), [go-no-go](#go-no-go), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/readysetgo.py)

___

### PerceptualDecisionMaking  
Doc: Two-alternative forced choice task in which the subject has to  
    integrate two stimuli to decide which one is higher on average.  
  
    A noisy stimulus is shown during the stimulus period. The strength (  
    coherence) of the stimulus is randomly sampled every trial. Because the  
    stimulus is noisy, the agent is encouraged to integrate the stimulus  
    over time.  
  
    Args:  
        cohs: list of float, coherence levels controlling the difficulty of  
            the task  
        sigma: float, input noise level  
        dim_ring: int, dimension of ring input and output  
      
Reference paper   
[The analysis of visual motion: a comparison of neuronal and psychophysical performance](https://www.jneurosci.org/content/12/12/4745)  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : 0.0  
  
Tags: [perceptual](#perceptual), [two-alternative](#two-alternative), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/perceptualdecisionmaking.py)

___

### PerceptualDecisionMakingDelayResponse  
Doc: Perceptual decision-making with delayed responses.  
  
    Agents have to integrate two stimuli and report which one is  
    larger on average after a delay.  
  
    Args:  
        stim_scale: Controls the difficulty of the experiment. (def: 1., float)  
      
Reference paper   
[Discrete attractor dynamics underlies persistent activity in the frontal cortex](https://www.nature.com/articles/s41586-019-0919-7)  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : 0.0  
  
Tags: [perceptual](#perceptual), [delayed response](#delayed-response), [two-alternative](#two-alternative), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/perceptualdecisionmaking.py)

___

### PostDecisionWager  
Doc: Post-decision wagering task assessing confidence.  
  
    The agent first performs a perceptual discrimination task (see for more  
    details the PerceptualDecisionMaking task). On a random half of the  
    trials, the agent is given the option to abort the sensory  
    discrimination and to choose instead a sure-bet option that guarantees a  
    small reward. Therefore, the agent is encouraged to choose the sure-bet  
    option when it is uncertain about its perceptual decision.  
      
Reference paper   
[Representation of Confidence Associated with a Decision by Neurons in the Parietal Cortex](https://science.sciencemag.org/content/324/5928/759.long)  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : 0.0  
sure : 0.7  
  
Tags: [perceptual](#perceptual), [delayed response](#delayed-response), [confidence](#confidence)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/postdecisionwager.py)

___

### ProbabilisticReasoning  
Doc: Probabilistic reasoning.  
  
    The agent is shown a sequence of stimuli. Each stimulus is associated  
    with a certain log-likelihood of the correct response being one choice  
    versus the other. The final log-likelihood of the target response being,  
    for example, option 1, is the sum of all log-likelihood associated with  
    the presented stimuli. A delay period separates each stimulus, so the  
    agent is encouraged to lean the log-likelihood association and integrate  
    these values over time within a trial.  
  
    Args:  
        shape_weight: array-like, evidence weight of each shape  
        n_loc: int, number of location of show shapes  
      
Reference paper   
[Probabilistic reasoning by neurons](https://www.nature.com/articles/nature05852)  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : 0.0  
  
Tags: [perceptual](#perceptual), [two-alternative](#two-alternative), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/probabilisticreasoning.py)

___

### PulseDecisionMaking  
Doc: Pulse-based decision making task.  
  
    Discrete stimuli are presented briefly as pulses.  
  
    Args:  
        p_pulse: array-like, probability of pulses for each choice  
        n_bin: int, number of stimulus bins  
      
Reference paper   
[Sources of noise during accumulation of evidence in unrestrained and voluntarily head-restrained rats](https://elifesciences.org/articles/11308)  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : 0.0  
  
Tags: [perceptual](#perceptual), [two-alternative](#two-alternative), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/perceptualdecisionmaking.py)

___

### Reaching1D  
Doc: Reaching to the stimulus.  
  
    The agent is shown a stimulus during the fixation period. The stimulus  
    encodes a one-dimensional variable such as a movement direction. At the  
    end of the fixation period, the agent needs to respond by reaching  
    towards the stimulus direction.  
      
Reference paper   
[Neuronal population coding of movement direction](https://science.sciencemag.org/content/233/4771/1416)  
  
Reward structure   
correct : 1.0  
fail : -0.1  
  
Tags: [motor](#motor), [steps action space](#steps-action-space)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/reaching.py)

___

### Reaching1DWithSelfDistraction  
Doc: Reaching with self distraction.  
  
    In this task, the reaching state itself generates strong inputs that  
    overshadows the actual target input. This task is inspired by behavior  
    in electric fish where the electric sensing organ is distracted by  
    discharges from its own electric organ for active sensing.  
    Similar phenomena in bats.  
      
Reference paper   
Missing paper name  
Missing paper link  
  
Reward structure   
correct : 1.0  
fail : -0.1  
  
Tags: [motor](#motor), [steps action space](#steps-action-space)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/reaching.py)

___

### ReachingDelayResponse  
Doc: Reaching task with a delay period.  
  
    A reaching direction is presented by the stimulus during the stimulus  
    period. Followed by a delay period, the agent needs to respond to the  
    direction of the stimulus during the decision period.  
      
Reference paper   
Missing paper name  
Missing paper link  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : -0.0  
miss : -0.5  
  
Tags: [perceptual](#perceptual), [delayed response](#delayed-response), [continuous action space](#continuous-action-space), [multidimensional action space](#multidimensional-action-space), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/reachingdelayresponse.py)

___

### ReadySetGo  
Doc: Agents have to measure and produce different time intervals.  
  
    A stimulus is briefly shown during a ready period, then again during a  
    set period. The ready and set periods are separated by a measure period,  
    the duration of which is randomly sampled on each trial. The agent is  
    required to produce a response after the set cue such that the interval  
    between the response and the set cue is as close as possible to the  
    duration of the measure period.  
  
    Args:  
        gain: Controls the measure that the agent has to produce. (def: 1, int)  
        prod_margin: controls the interval around the ground truth production  
            time within which the agent receives proportional reward  
      
Reference paper   
[Flexible Sensorimotor Computations through Rapid Reconfiguration of Cortical Dynamics](https://www.sciencedirect.com/science/article/pii/S0896627318304185)  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : 0.0  
  
Tags: [timing](#timing), [go-no-go](#go-no-go), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/readysetgo.py)

___

### SingleContextDecisionMaking  
Doc: Context-dependent decision-making task.  
  
    The agent simultaneously receives stimulus inputs from two modalities (  
    for example, a colored random dot motion pattern with color and motion  
    modalities). The agent needs to make a perceptual decision based on only  
    one of the two modalities, while ignoring the other. The agent reports  
    its decision during the decision period, with an optional delay period  
    in between the stimulus period and the decision period. The relevant  
    modality is not explicitly signaled.  
  
    Args:  
        context: int, 0 or 1 for the two context (rules). If 0, need to  
            focus on modality 0 (the first one)  
      
Reference paper   
[Context-dependent computation by recurrent dynamics in prefrontal cortex](https://www.nature.com/articles/nature12742)  
  
Reward structure   
abort : -0.1  
correct : 1.0  
  
Tags: [perceptual](#perceptual), [context dependent](#context-dependent), [two-alternative](#two-alternative), [supervised](#supervised)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/contextdecisionmaking.py)

___

### SpatialSuppressMotion  
Doc:   
    Spatial suppression motion task. This task is useful to study center-surround interaction in monkey MT and human psychophysical performance in motion perception.  
  
    Tha task is derived from (Tadin et al. Nature, 2003). In this task, there is no fixation or decision stage. We only present a stimulus and a subject needs to perform a 4-AFC motion direction judgement. The ground-truth is the probabilities for choosing the four directions at a given time point. The probabilities depend on stimulus contrast and size, and the probabilities are derived from emprically measured human psychophysical performance.  
  
    In this version, the input size is 4 (directions) x 8 (size) = 32 neurons. This setting aims to simulate four pools (8 neurons in each pool) of neurons that are selective for four directions.   
  
    Args:  
        <dt>: millisecs per image frame, default: 8.3 (given 120HZ monitor)  
        <win_size>: size per image frame  
        <timing>: millisecs, stimulus duration, default: 8.3 * 36 frames ~ 300 ms.   
            This is the longest duration we need (i.e., probability reach ceilling)  
      
    Note that please input default seq_len = 36 frames when creating dataset object.  
  
  
      
Reference paper   
[Perceptual consequences of centre–surround antagonism in visual motion processing](https://www.nature.com/articles/nature01800)  
  
Reward structure   
abort : -0.1  
correct : 1.0  
fail : 0.0  
  
Tags: [perceptual](#perceptual), [plaid](#plaid), [motion](#motion), [center-surround](#center-surround)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/envs/spatialsuppressmotion.py)

___

### Tags ### 

### confidence 

[PostDecisionWager-v0](#postdecisionwager)

### context dependent 

[ContextDecisionMaking-v0](#contextdecisionmaking)

[SingleContextDecisionMaking-v0](#singlecontextdecisionmaking)

### continuous action space 

[ReachingDelayResponse-v0](#reachingdelayresponse)

### delayed response 

[GoNogo-v0](#gonogo)

[IntervalDiscrimination-v0](#intervaldiscrimination)

[PerceptualDecisionMakingDelayResponse-v0](#perceptualdecisionmakingdelayresponse)

[PostDecisionWager-v0](#postdecisionwager)

[ReachingDelayResponse-v0](#reachingdelayresponse)

### go-no-go 

[DelayPairedAssociation-v0](#delaypairedassociation)

[GoNogo-v0](#gonogo)

[MotorTiming-v0](#motortiming)

[OneTwoThreeGo-v0](#onetwothreego)

[ReadySetGo-v0](#readysetgo)

### motor 

[Reaching1D-v0](#reaching1d)

[Reaching1DWithSelfDistraction-v0](#reaching1dwithselfdistraction)

### multidimensional action space 

[ReachingDelayResponse-v0](#reachingdelayresponse)

### n-alternative 

[Bandit-v0](#bandit)

### perceptual 

[AntiReach-v0](#antireach)

[ContextDecisionMaking-v0](#contextdecisionmaking)

[DelayComparison-v0](#delaycomparison)

[DelayMatchCategory-v0](#delaymatchcategory)

[DelayMatchSample-v0](#delaymatchsample)

[DelayMatchSampleDistractor1D-v0](#delaymatchsampledistractor1d)

[DelayPairedAssociation-v0](#delaypairedassociation)

[DualDelayMatchSample-v0](#dualdelaymatchsample)

[EconomicDecisionMaking-v0](#economicdecisionmaking)

[HierarchicalReasoning-v0](#hierarchicalreasoning)

[MultiSensoryIntegration-v0](#multisensoryintegration)

[PerceptualDecisionMaking-v0](#perceptualdecisionmaking)

[PerceptualDecisionMakingDelayResponse-v0](#perceptualdecisionmakingdelayresponse)

[PostDecisionWager-v0](#postdecisionwager)

[ProbabilisticReasoning-v0](#probabilisticreasoning)

[PulseDecisionMaking-v0](#pulsedecisionmaking)

[ReachingDelayResponse-v0](#reachingdelayresponse)

[SingleContextDecisionMaking-v0](#singlecontextdecisionmaking)

[SpatialSuppressMotion-v0](#spatialsuppressmotion)

### reaction time 

### steps action space 

[AntiReach-v0](#antireach)

[Reaching1D-v0](#reaching1d)

[Reaching1DWithSelfDistraction-v0](#reaching1dwithselfdistraction)

### supervised 

[ContextDecisionMaking-v0](#contextdecisionmaking)

[DelayComparison-v0](#delaycomparison)

[DelayMatchCategory-v0](#delaymatchcategory)

[DelayMatchSample-v0](#delaymatchsample)

[DelayMatchSampleDistractor1D-v0](#delaymatchsampledistractor1d)

[DelayPairedAssociation-v0](#delaypairedassociation)

[DualDelayMatchSample-v0](#dualdelaymatchsample)

[GoNogo-v0](#gonogo)

[HierarchicalReasoning-v0](#hierarchicalreasoning)

[IntervalDiscrimination-v0](#intervaldiscrimination)

[MotorTiming-v0](#motortiming)

[MultiSensoryIntegration-v0](#multisensoryintegration)

[OneTwoThreeGo-v0](#onetwothreego)

[PerceptualDecisionMaking-v0](#perceptualdecisionmaking)

[PerceptualDecisionMakingDelayResponse-v0](#perceptualdecisionmakingdelayresponse)

[ProbabilisticReasoning-v0](#probabilisticreasoning)

[PulseDecisionMaking-v0](#pulsedecisionmaking)

[ReachingDelayResponse-v0](#reachingdelayresponse)

[ReadySetGo-v0](#readysetgo)

[SingleContextDecisionMaking-v0](#singlecontextdecisionmaking)

### timing 

[IntervalDiscrimination-v0](#intervaldiscrimination)

[MotorTiming-v0](#motortiming)

[OneTwoThreeGo-v0](#onetwothreego)

[ReadySetGo-v0](#readysetgo)

### two-alternative 

[ContextDecisionMaking-v0](#contextdecisionmaking)

[DawTwoStep-v0](#dawtwostep)

[DelayComparison-v0](#delaycomparison)

[DelayMatchCategory-v0](#delaymatchcategory)

[DelayMatchSample-v0](#delaymatchsample)

[DelayMatchSampleDistractor1D-v0](#delaymatchsampledistractor1d)

[DualDelayMatchSample-v0](#dualdelaymatchsample)

[HierarchicalReasoning-v0](#hierarchicalreasoning)

[IntervalDiscrimination-v0](#intervaldiscrimination)

[MultiSensoryIntegration-v0](#multisensoryintegration)

[PerceptualDecisionMaking-v0](#perceptualdecisionmaking)

[PerceptualDecisionMakingDelayResponse-v0](#perceptualdecisionmakingdelayresponse)

[ProbabilisticReasoning-v0](#probabilisticreasoning)

[PulseDecisionMaking-v0](#pulsedecisionmaking)

[SingleContextDecisionMaking-v0](#singlecontextdecisionmaking)

### value-based 

[EconomicDecisionMaking-v0](#economicdecisionmaking)

### working memory 

[DelayComparison-v0](#delaycomparison)

[DelayMatchCategory-v0](#delaymatchcategory)

[DelayMatchSample-v0](#delaymatchsample)

[DelayMatchSampleDistractor1D-v0](#delaymatchsampledistractor1d)

[DelayPairedAssociation-v0](#delaypairedassociation)

[DualDelayMatchSample-v0](#dualdelaymatchsample)

[IntervalDiscrimination-v0](#intervaldiscrimination)

