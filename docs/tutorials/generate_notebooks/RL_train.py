#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 09:33:08 2020

@author: manuel
"""

import os
from pathlib import Path
import json
import importlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from neurogym.wrappers import ALL_WRAPPERS
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.callbacks import CheckpointCallback
import gym
import glob
import neurogym as ngym


def get_modelpath(envid):
    # Make a local file directories
    path = Path('.') / 'files'
    os.makedirs(path, exist_ok=True)
    path = path / envid
    os.makedirs(path, exist_ok=True)
    return path


def apply_wrapper(env, wrap_string, params):
    wrap_str = ALL_WRAPPERS[wrap_string]
    wrap_module = importlib.import_module(wrap_str.split(":")[0])
    wrap_method = getattr(wrap_module, wrap_str.split(":")[1])
    return wrap_method(env, **params)


def make_env(env_id, rank, seed=0, wrapps={}, **kwargs):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    """
    def _init():
        env = gym.make(env_id, **kwargs)
        env.seed(seed + rank)
        for wrap in wrapps.keys():
            if not (wrap == 'MonitorExtended-v0' and rank != 0):
                env = apply_wrapper(env, wrap, wrapps[wrap])
        return env
    set_global_seeds(seed)
    return _init


def get_alg(alg):
    if alg == "A2C":
        from stable_baselines import A2C as algo
    elif alg == "ACER":
        from stable_baselines import ACER as algo
    elif alg == "ACKTR":
        from stable_baselines import ACKTR as algo
    elif alg == "PPO2":
        from stable_baselines import PPO2 as algo
    return algo


def train_network(envid):
    """Supervised training networks.

    Save network in a path determined by environment ID.

    Args:
        envid: str, environment ID.
    """

    modelpath = get_modelpath(envid)
    config = {
        'dt': 100,
        'hidden_size': 64,
        'lr': 1e-2,
        'alg': 'ACER',
        'rollout': 20,
        'n_thrds': 1,
        'wrappers_kwargs': {},
        'alg_kwargs': {},
        'seed': 0,
        # 'num_steps': 100000,
        'num_steps': 100,
        'envid': envid,
    }

    env_kwargs = {'dt': config['dt']}
    config['env_kwargs'] = env_kwargs

    # Save config
    with open(modelpath / 'config.json', 'w') as f:
        json.dump(config, f)
    algo = get_alg(config['alg'])
    # Make supervised dataset
    make_envs = [make_env(env_id=envid, rank=i, seed=config['seed'],
                                  wrapps=config['wrappers_kwargs'],
                                  **env_kwargs)
                         for i in range(config['n_thrds'])]
    # env = SubprocVecEnv(make_envs)
    env = DummyVecEnv(make_envs)  # Less efficient but more robust
    model = algo(LstmPolicy, env, verbose=0, n_steps=config['rollout'],
                 n_cpu_tf_sess=config['n_thrds'], tensorboard_log=None,
                 policy_kwargs={"feature_extraction": "mlp",
                                "n_lstm": config['hidden_size']},
                 **config['alg_kwargs'])
    chckpnt_cllbck = CheckpointCallback(save_freq=int(config['num_steps']/10),
                                        save_path=modelpath,
                                        name_prefix='model')
    model.learn(total_timesteps=config['num_steps'], callback=chckpnt_cllbck)
    print('Finished Training')


def infer_test_timing(env):
    """Infer timing of environment for testing."""
    timing = {}
    for period in env.timing.keys():
        period_times = [env.sample_time(period) for _ in range(100)]
        timing[period] = np.median(period_times)
    return timing


def extend_obs(ob, num_threads):
    sh = ob.shape
    return np.concatenate((ob, np.zeros((num_threads-sh[0], sh[1]))))


def order_by_sufix(file_list):
    file_list = [os.path.basename(x) for x in file_list]
    flag = 'model.zip' in file_list
    file_list = [x for x in file_list if x != 'model.zip']
    sfx = [int(x[x.find('_')+1:x.rfind('_')]) for x in file_list]
    sorted_list = [x for _, x in sorted(zip(sfx, file_list))]
    if flag:
        sorted_list.append('model.zip')
    return sorted_list, np.max(sfx)


def run_network(envid):
    """Run trained networks for analysis.

    Args:
        envid: str, Environment ID

    Returns:
        activity: a list of activity matrices
        info: pandas dataframe, each row is information of a trial
        config: dict of network, training configurations
    """
    modelpath = get_modelpath(envid)
    files = glob.glob(modelpath)
    # files = glob.glob(str(modelpath)+'/model*')
    if len(files) > 0:
        with open(modelpath / 'config.json') as f:
            config = json.load(f)
        env_kwargs = config['env_kwargs']
        wrappers_kwargs = config['wrappers_kwargs']
        seed = config['seed']
        # Run network to get activity and info
        sorted_models, last_model = order_by_sufix(files)
        model_name = sorted_models[-1]
        algo = get_alg(config['alg'])
        model = algo.load(modelpath / model_name, tensorboard_log=None,
                          custom_objects={'verbose': 0})

        # Environment
        env = make_env(env_id=envid, rank=0, seed=seed, wrapps=wrappers_kwargs,
                       **env_kwargs)()
        env.timing = infer_test_timing(env)
        env.reset(no_step=True)
        # Instantiate the network and print information
        activity = list()
        state_mat = []
        ob = env.reset()
        _states = None
        done = False
        info_df = pd.DataFrame()
        # num_steps = 10 ** 5
        num_steps = 10 ** 3
        for stp in range(int(num_steps)):
            ob = np.reshape(ob, (1, ob.shape[0]))
            done = [done] + [False for _ in range(config['n_thrds']-1)]
            action, _states = model.predict(extend_obs(ob, config['n_thrds']),
                                            state=_states, mask=done)
            action = action[0]
            ob, rew, done, info = env.step(action)
            if done:
                env.reset()
            if isinstance(info, (tuple, list)):
                info = info[0]
                action = action[0]
            state_mat.append(_states[0, int(_states.shape[1]/2):])
            if info['new_trial']:
                gt = env.gt_now
                correct = action == gt
                # Log trial info
                trial_info = env.trial
                trial_info.update({'correct': correct, 'choice': action})
                info_df = info_df.append(trial_info, ignore_index=True)
                # Log stimulus period activity
                state_mat = np.array(state_mat)
                
                # Excluding decision period if exists
                if 'decision' in env.start_ind:
                    state_mat = state_mat[:env.start_ind['decision']]
                
                activity.append(state_mat)
                state_mat = []
        env.close()

        activity = np.array(activity)
        info = info_df
        return activity, info, config


if __name__ == '__main__':
    envid = 'PerceptualDecisionMaking-v0'
    # train_network(envid)
    activity, info, config = run_network(envid)
    print(len(activity))
    print(activity[0].shape)
    print(info)
    print(config)
