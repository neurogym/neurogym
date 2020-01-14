### Bandit

Original paper: 

Missing paper name

Link: Missing paper link

### DPA

Original paper: 

Active information maintenance in working memory by a sensory cortex

Link: https://elifesciences.org/articles/43191

Epochs 

fixation : truncated_exponential [500, 200, 800]

stim1 : truncated_exponential [500, 200, 800]

delay_btw_stim : truncated_exponential [500, 200, 800]

stim2 : truncated_exponential [500, 200, 800]

delay_aft_stim : truncated_exponential [500, 200, 800]

decision : truncated_exponential [500, 200, 800]

### DawTwoStep

Original paper: 

Model-Based Influences on Humans' 
        Choices and Striatal Prediction Errors

Link: https://www.sciencedirect.com/science/article/pii/S0896627311001255

### DelayedMatchCategory

Original paper: 

Experience-dependent representation
        of visual categories in parietal cortex

Link: https://www.nature.com/articles/nature05078

Epochs 

fixation : constant 500

sample : constant 500

delay : constant 1500

test : constant 500

decision : constant 500

### DelayedMatchToSample

Original paper: 

Neural Mechanisms of Visual Working Memory 
        in Prefrontal Cortex of the Macaque

Link: https://www.jneurosci.org/content/jneuro/16/16/5154.full.pdf

Epochs 

fixation : constant 500

sample : constant 500

delay : constant 1500

test : constant 500

decision : constant 500

### DR

Original paper: 

Missing paper name

Link: Missing paper link

Epochs 

fixation : constant 500

stimulus : truncated_exponential [330, 80, 1500]

delay : choice [1000, 5000, 10000]

decision : constant 500

### GNG

Original paper: 

Active information maintenance in working memory by a sensory cortex

Link: https://elifesciences.org/articles/43191

Epochs 

fixation : constant 100

stimulus : constant 200

resp_delay : constant 100

decision : constant 100

### GenTask

Original paper: 

Missing paper name

Link: Missing paper link

Epochs 

fixation : truncated_exponential [500, 200, 800]

stim1 : truncated_exponential [500, 200, 800]

delay_btw_stim : truncated_exponential [500, 200, 800]

stim2 : truncated_exponential [500, 200, 800]

delay_aft_stim : truncated_exponential [500, 200, 800]

decision : truncated_exponential [500, 200, 800]

### IBL

Original paper: 

An International Laboratory for Systems and Computational Neuroscience

Link: https://www.sciencedirect.com/science/article/pii/S0896627317311364

### Mante

Original paper: 

Context-dependent computation by recurrent 
        dynamics in prefrontal cortex

Link: https://www.nature.com/articles/nature12742

Epochs 

fixation : constant 750

stimulus : constant 750

delay : truncated_exponential [300, 83, 1200]

decision : constant 500

### MatchingPenny

Original paper: 

Prefrontal cortex and decision making in a mixed-strategy game

Link: https://www.nature.com/articles/nn1209

### MemoryRecall

Original paper: 

Missing paper name

Link: Missing paper link

### nalt_RDM

Original paper: 

Missing paper name

Link: Missing paper link

Epochs 

fixation : constant 500

stimulus : truncated_exponential [330, 80, 1500]

decision : constant 500

### RDM

Original paper: 

Bounded Integration in Parietal Cortex Underlies
        Decisions Even When Viewing Duration Is Dictated by the Environment

Link: http://www.jneurosci.org/content/28/12/3017

Epochs 

fixation : constant 500

stimulus : truncated_exponential [330, 80, 1500]

decision : constant 500

### ReadySetGo

Original paper: 

Flexible Sensorimotor Computations through Rapid
        Reconfiguration of Cortical Dynamics

Link: https://www.sciencedirect.com/science/article/pii/S0896627318304185

Epochs 

fixation : constant 500

ready : constant 83

measure : choice [500, 580, 660, 760, 840, 920, 1000]

set : constant 83

### Romo

Original paper: 

Neuronal Population Coding of Parametric Working Memory

Link: https://www.jneurosci.org/content/30/28/9424

Epochs 

fixation : constant 750

f1 : constant 500

delay : truncated_exponential [3000, 2700, 3300]

f2 : constant 500

decision : constant 500

### PadoaSch

Original paper: 

Neurons in the orbitofrontal cortex encode economic value

Link: https://www.nature.com/articles/nature04676

Epochs 

fixation : constant 750

offer_on : truncated_exponential [1500, 1000, 2000]

decision : constant 750

### PDWager

Original paper: 

Representation of Confidence Associated with a
         Decision by Neurons in the Parietal Cortex

Link: https://science.sciencemag.org/content/324/5928/759.long

Epochs 

fixation : constant 750

stimulus : truncated_exponential [180, 100, 800]

delay : truncated_exponential [1350, 1200, 1800]

pre_sure : truncated_exponential [575, 500, 750]

decision : constant 500

