* Under development, details subject to change

### List of 11 wrappers implemented

[CatchTrials-v0](#catchtrials-v0)

[Combine-v0](#combine-v0)

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
  
Logic: Introduces catch trials in which the reward for a correct choice is modified (e.g. is set to the reward for an incorrect choice). Note that the wrapper only changes the reward associated to a correct answer and does not change the ground truth. Thus, the catch trial affect a pure supervised learning setting.  
  
[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/wrappers/catch_trials.py)

___

### Combine-v0  
  
Logic: Allows to combine two tasks, one of which working as the distractor task.  
  
Reference paper   
  
[Response outcomes gate the impact of expectations on perceptual decisions](https://www.biorxiv.org/content/10.1101/433409v3)  
  
[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/wrappers/combine.py)

___

### MissTrialReward-v0  
  
Logic: Add a negative reward if a trial ends with no action.  
  
[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/wrappers/miss_trials_reward.py)

___

### Monitor-v0  
  
Logic: Saves relevant behavioral information: rewards, actions, observations, new trial, ground truth.  
  
[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/wrappers/monitor.py)

___

### Noise-v0  
  
Logic: Add Gaussian noise to the observations.  
  
[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/wrappers/noise.py)

___

### PassAction-v0  
  
Logic: Modifies observation by adding the previous action.  
  
[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/wrappers/pass_action.py)

___

### PassReward-v0  
  
Logic: Modifies observation by adding the previous reward.  
  
[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/wrappers/pass_reward.py)

___

### ReactionTime-v0  
  
Logic: Modifies a given environment by allowing the network to act at any time after the fixation period.  
  
[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/wrappers/reaction_time.py)

___

### SideBias-v0  
  
Logic: Changes the probability of ground truth.  
  
[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/wrappers/side_bias.py)

___

### TTLPulse-v0  
  
Logic: Outputs extra pulses that will be non-zero during specified periods.  
  
[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/wrappers/ttl_pulse.py)

___

### TrialHistory-v0  
  
Logic: Change ground truth probability based on previousoutcome.  
  
Reference paper   
  
[Response outcomes gate the impact of expectations on perceptual decisions](https://www.biorxiv.org/content/10.1101/433409v3)  
  
[Source](https://github.com/gyyang/neurogym/blob/master/neurogym/wrappers/trial_hist.py)

