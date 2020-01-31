* Under development, details subject to change

### List of wrappers implemented

* 10 wrappers implemented so far.

[CatchTrials-v0](#catchtrials-v0)

[MissTrialReward-v0](#misstrialreward-v0)

[Monitor-v0](#monitor-v0)

[Noise-v0](#noise-v0)

[PassAction-v0](#passaction-v0)

[PassReward-v0](#passreward-v0)

[ReactionTime-v0](#reactiontime-v0)

[SideBias-v0](#sidebias-v0)

[TTLPulse-v0](#ttlpulse-v0)

[TrialHistory-v0](#trialhistory-v0)

___

### CatchTrials-v0

Logic: Introduces catch trials in which the reward for
         a correct choice is modified (e.g. is set to the reward for an
         incorrect choice). Note that the wrapper only changes the reward
         associated to a correct answer and does not change the ground truth.
         Thus, the catch trial affect a pure supervised learning setting.

Input parameters: 

stim_th : Percentile of stimulus distribution below which catch
        trials are allowed (in some cases, experimenter might decide not
        to have catch trials when  stimulus is very obvious). (def: 50)

start : Number of trials after which the catch trials can occur.
        (def: 0)

catch_prob : Catch trial probability. (def: 0.1)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/wrappers/catch_trials.py)

___

### MissTrialReward-v0

Logic: Add a negative reward if a trial ends with no action.

Input parameters: 

r_miss : Reward given when a miss trial occurs.(def: 0)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/wrappers/miss_trials_reward.py)

___

### Monitor-v0

Logic: Saves relevant behavioral information: rewards, actions, observations, new trial, ground truth.

Input parameters: 

folder : Folder where the data will be saved. (def: None)

verbose : Whether to print information about average reward and number of trials

num_tr_save : Data will be saved every num_tr_save trials.
        (def: 100000)

sv_fig : Whether to save a figure of the experiment structure. If True, a figure will be updated every num_tr_save. (def: False)

num_stps_sv_fig : Number of trial steps to include in the figure. (def: 100)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/wrappers/monitor.py)

___

### Noise-v0

Logic: Add Gaussian noise to the observations.

Input parameters: 

std_noise : Standard deviation of noise. (def: 0.1)

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/wrappers/noise.py)

___

### PassAction-v0

Logic: Modifies observation by adding the previous
        action.

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/wrappers/pass_action.py)

___

### PassReward-v0

Logic: Modifies observation by adding the previous
        reward.

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/wrappers/pass_reward.py)

___

### ReactionTime-v0

Logic: Modfies a given environment by allowing the network
        to act at any time after the fixation period.

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/wrappers/reaction_time.py)

___

### SideBias-v0

Logic: Changes the probability of ground truth.

Input parameters: 

block_dur : Number of trials per block. (def: 200 (int))

prob : Specifies probabilities for each choice. Within each block,
        the probability should sum up to 1.
        (def: None (Numpy array (n_block, n_choices)))

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/wrappers/side_bias.py)

___

### TTLPulse-v0

Logic: Outputs extra pulses that will be non-zero during specified periods.

Input parameters: 

periods : List of list specifying the on periods for each pulse. (def: [])

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/wrappers/ttl_pulse.py)

___

### TrialHistory-v0

Logic: Change ground truth probability based on previous outcome.

Reference paper: 

[Response outcomes gate the impact of expectations
         on perceptual decisions](https://www.biorxiv.org/content/10.1101/433409v3)

Input parameters: 

blk_ch_prob : If not None, specifies the probability of changing
        block (randomly). (def: None)

block_dur : Number of trials per block. (def: 200 (int))

rep_prob : Specifies probabilities of repeating for each block.
        (def: (.2, .8))

[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/wrappers/trial_hist.py)

