import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import gym
import neurogym as ngym


def analysis_average_activity(activity, info, config):
    # Load and preprocess results
    plt.figure(figsize=(1.2, 0.8))
    t_plot = np.arange(activity.shape[1]) * config['dt']
    plt.plot(t_plot, activity.mean(axis=0).mean(axis=-1))

    # os.makedirs(FIGUREPATH / path, exist_ok=True)
    # plt.savefig(FIGUREPATH / path / 'meanactivity.png', transparent=True, dpi=600)


def get_conditions(info):
    """Get a list of task conditions to plot."""
    conditions = info.columns
    # This condition's unique value should be less than 5
    new_conditions = list()
    for c in conditions:
        try:
            n_cond = len(pd.unique(info[c]))
            if 1 < n_cond < 5:
                new_conditions.append(c)
        except TypeError:
            pass
        
    return new_conditions


def analysis_activity_by_condition(activity, info, config):
    conditions = get_conditions(info)
    for condition in conditions:
        values = pd.unique(info[condition])
        plt.figure(figsize=(1.2, 0.8))
        t_plot = np.arange(activity.shape[1]) * config['dt']
        for value in values:
            a = activity[info[condition] == value]
            plt.plot(t_plot, a.mean(axis=0).mean(axis=-1), label=str(value))
        plt.legend(title=condition, loc='center left', bbox_to_anchor=(1.0, 0.5))
        
        # os.makedirs(FIGUREPATH / path, exist_ok=True)
        

def analysis_example_units_by_condition(activity, info, config):
    conditions = get_conditions(info)
    if len(conditions) < 1:
        return

    example_ids = np.array([0, 1])    
    for example_id in example_ids:        
        example_activity = activity[:, :, example_id]
        fig, axes = plt.subplots(
                len(conditions), 1,  figsize=(1.2, 0.8 * len(conditions)),
                sharex=True)
        for i, condition in enumerate(conditions):
            ax = axes[i]
            values = pd.unique(info[condition])
            t_plot = np.arange(activity.shape[1]) * config['dt']
            for value in values:
                a = example_activity[info[condition] == value]
                ax.plot(t_plot, a.mean(axis=0), label=str(value))
            ax.legend(title=condition, loc='center left', bbox_to_anchor=(1.0, 0.5))
            ax.set_ylabel('Activity')
            if i == len(conditions) - 1:
                ax.set_xlabel('Time (ms)')
            if i == 0:
                ax.set_title('Unit {:d}'.format(example_id + 1))


def analysis_pca_by_condition(activity, info, config):
    # Reshape activity to (N_trial x N_time, N_neuron)
    activity_reshape = np.reshape(activity, (-1, activity.shape[-1]))
    pca = PCA(n_components=2)
    pca.fit(activity_reshape)
    
    conditions = get_conditions(info)
    for condition in conditions:
        values = pd.unique(info[condition])
        fig = plt.figure(figsize=(2.5, 2.5))
        ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
        for value in values:
            # Get relevant trials, and average across them
            a = activity[info[condition] == value].mean(axis=0)
            a = pca.transform(a)  # (N_time, N_PC)
            plt.plot(a[:, 0], a[:, 1], label=str(value))
        plt.legend(title=condition, loc='center left', bbox_to_anchor=(1.0, 0.5))
    
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
