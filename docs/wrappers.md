* Under development, details subject to change

### List of wrappers implemented

* 9 wrappers implemented so far.

[CatchTrials-v0](#catchtrials-v0)

[MissTrialReward-v0](#misstrialreward-v0)

[Monitor-v0](#monitor-v0)

[Noise-v0](#noise-v0)

[PassAction-v0](#passaction-v0)

[PassReward-v0](#passreward-v0)

[ReactionTime-v0](#reactiontime-v0)

[SideBias-v0](#sidebias-v0)

[TrialHistory-v0](#trialhistory-v0)

### CatchTrials-v0

Logic: Introduces catch trials in which the reward for
         a correct choice is modified (e.g. is set to the reward for an
         incorrect choice). Note that the wrapper only changes the reward
         associated to a correct answer and does not change the ground truth.
         Thus, the catch trial affect a pure supervised learning setting.

Input parameters: 

catch_prob : Catch trial probability. (def: 0.1)

stim_th : Percentile of stimulus distribution below which catch
        trials are allowed (in some cases, experimenter might decide not
        to have catch trials when  stimulus is very obvious). (def: 50)

start : Number of trials after which the catch trials can occur.
        (def: 0)

### MissTrialReward-v0

Logic: Add a negative reward if a trial ends with no action.

Input parameters: 

r_miss : Reward given when a miss trial occurs.(def: 0)

### Monitor-v0

Logic: Saves relevant behavioral information: rewards, actions, observations, new trial, ground truth.

Input parameters: 

num_tr_save : Data will be saved every num_tr_save trials.
        (def: 100000)

info_keywords : (tuple) extra information to log, from the information return of environment.step

verbose : Whether to print information about average reward and number of trials

folder : Folder where the data will be saved. (def: None)

### Noise-v0

Logic: Add Gaussian noise to the observations.

Input parameters: 

std_noise : Standard deviation of noise. (def: 0.1)

### PassAction-v0

Logic: Modifies observation by adding the previous
        action.

### PassReward-v0

Logic: Modifies observation by adding the previous
        reward.

### ReactionTime-v0

Logic: Modfies a given environment by allowing the network
        to act at any time after the fixation period.

### SideBias-v0

Logic: Changes the probability of ground truth.

Input parameters: 

block_dur : Number of trials per block. (def: 200 (int))

prob : Specifies probabilities for each choice. Within each block,
        the probability should sum up to 1.
        (def: None (Numpy array (n_block, n_choices)))

### TrialHistory-v0

Logic: Change ground truth probability based on previous outcome.

Reference paper: 

[Response outcomes gate the impact of expectations
         on perceptual decisions](https://www.biorxiv.org/content/10.1101/433409v3)

Input parameters: 

block_dur : Number of trials per block. (def: 200 (int))

rep_prob : Specifies probabilities of repeating for each block.
        (def: (.2, .8))

blk_ch_prob : If not None, specifies the probability of changing
        block (randomly). (def: None)

