"""Created on Tue Aug 18 17:16:10 2020.

@author: molano

Usage:

    import neurogym as ngym

    import gymnasium as gym
    kwargs = {'dt': 100, 'tr_hist_kwargs': {'probs': 0.9}}
    # Make supervised dataset
    tasks = ngym.get_collection('priors')
    envs = [gym.make(task, **kwargs) for task in tasks]

"""

import gymnasium as gym

from neurogym import wrappers


def priors_v0(tr_hist_kwargs=None, var_nch_kwargs=None, **task_kwargs):
    if var_nch_kwargs is None:
        var_nch_kwargs = {}
    if tr_hist_kwargs is None:
        tr_hist_kwargs = {"probs": 0.9}
    env = gym.make("NAltPerceptualDecisionMaking-v0", **task_kwargs)
    print(tr_hist_kwargs)
    env = wrappers.TrialHistoryEvolution(env, **tr_hist_kwargs)
    env = wrappers.Variable_nch(env, **var_nch_kwargs)
    env = wrappers.PassAction(env)
    return wrappers.PassReward(env)
