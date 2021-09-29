* Under development, details subject to change

### List of 6 wrappers implemented

[Monitor-v0](#monitor-v0)

[Noise-v0](#noise-v0)

[PassAction-v0](#passaction-v0)

[PassReward-v0](#passreward-v0)

[ReactionTime-v0](#reactiontime-v0)

[SideBias-v0](#sidebias-v0)

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

