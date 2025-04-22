import os
import pickle
import sys
import time
from itertools import compress
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import optimize, stats
from sklearn.metrics import (accuracy_score, auc, confusion_matrix,
                             roc_auc_score, roc_curve)

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

def main():
    num_neurons = 150
    n_boot = 1000
    threshold = 0.5
    th = 0.2

    n_splits = 2
    n_trials = 8192

    data_path = '/Users/lexotto/Documents_Mac/Stage/UVA/Code/BehavioralVariabilityRNN-main/data/good_runs'
    networks = os.listdir(data_path)
    networks = list(filter(lambda x: x.isnumeric(), networks))
    networks.sort(key=lambda x: int(x))

    for network in networks:
        netw = str(network)
        print(f'Started network {network}')
        start_time = time.time()
        with open(os.path.join(data_path, netw, 'test_trials.pkl'), 'rb') as f:
                trials = pickle.load(f)

        net_out = np.load(os.path.join(data_path, netw, 'test_output.npy'))

        rnn_output = np.zeros((n_trials, 551, 150))
        for i in range(n_splits):
            rnn_output_temp = np.load(os.path.join(data_path, netw, f'batch2/rnn_output_p{i+1}.npy'))
            rnn_output[int(i * n_trials / n_splits) :int((i+1)* n_trials / n_splits)] = rnn_output_temp

        out_diff = net_out[:, trials['phases']['stimulus'], 1] - net_out[:, trials['phases']['stimulus'], 0]

        print(f'first check')

        # time step when network made the decision
        decision_time = np.argmax(np.abs(out_diff) > th, axis=1)

        out_diff_onset_stimulus = net_out[:, trials['phases']['stimulus'][0], 1] - net_out[:, trials['phases']['stimulus'][0], 0]

        #Difference in output is less than threshold at the start
        analysed_trials_valid_start = np.nonzero(np.abs(out_diff_onset_stimulus) <= th)[0]

        #A decision is made
        analysed_trials_choice_made = np.nonzero(np.sum(np.abs(out_diff) > th, axis=1) != 0)[0]

        #Both a valid start and a choice made
        analysed_trials_good_start_choice_made = np.intersect1d(analysed_trials_valid_start, analysed_trials_choice_made)

        # predicted choice
        choice = (out_diff[analysed_trials_good_start_choice_made, decision_time[analysed_trials_good_start_choice_made]] > 0).astype(np.int_)

        net_out = np.load(os.path.join(data_path, netw, 'test_output.npy'))

        correct_choices = np.nonzero(choice == trials['choice'][analysed_trials_good_start_choice_made])[0]
        correct_trials = analysed_trials_good_start_choice_made[correct_choices]


        choose_1 = choice[correct_choices] == 1
        choose_2 = choice[correct_choices] == 0

        choose_1_trials = correct_trials[choose_1]
        choose_2_trials = correct_trials[choose_2]

        unisensory_trials = np.where(np.logical_or(trials['modality'] == 'v', trials['modality'] == 'a'))[0]
        unisensory_trials_with_decision_1 = np.intersect1d(choose_1_trials, unisensory_trials)
        unisensory_trials_with_decision_2 = np.intersect1d(choose_2_trials, unisensory_trials)

        visual_trial_decision_1 = trials['modality'][unisensory_trials_with_decision_1] == 'v'
        visual_trial_decision_2 = trials['modality'][unisensory_trials_with_decision_2] == 'v'

        #For modality
        auc_neurons_mod = np.zeros((num_neurons, 4)) #1:dec 1 auc, #2 dec 2 auc, #3: dec1 -1, 0, 1 #4: dec2 -1, 0, 1

        print(f'second check')

        for neuron in range(num_neurons):
            if neuron == 50 or neuron == 100 or neuron == 150:
                print(f"Started neuron {neuron}")
            test_output_dec_1 = np.mean(rnn_output[unisensory_trials_with_decision_1, -20:, neuron], axis=1)
            test_output_dec_2 = np.mean(rnn_output[unisensory_trials_with_decision_2, -20:, neuron], axis=1)

            fpr_dec1, tpr_dec1, thresholds_dec1 = roc_curve(visual_trial_decision_1, test_output_dec_1)
            fpr_dec2, tpr_dec2, thresholds_dec2 = roc_curve(visual_trial_decision_2, test_output_dec_2)

            auc_neuron_dec1 = auc(fpr_dec1,tpr_dec1)
            auc_neuron_dec2 = auc(fpr_dec2,tpr_dec2)

            auc_neurons_mod[neuron, 0] = auc_neuron_dec1
            auc_neurons_mod[neuron, 1] = auc_neuron_dec2

            auc_boot_dec1 = np.zeros(n_boot)
            auc_boot_dec2 = np.zeros(n_boot)

            for i in range(n_boot):
                permuted_choices_dec1 = np.random.permutation(visual_trial_decision_1)
                permuted_choices_dec2 = np.random.permutation(visual_trial_decision_2)

                fpr_perm_dec1, tpr_perm_dec1, _ = roc_curve(permuted_choices_dec1, test_output_dec_1)
                fpr_perm_dec2, tpr_perm_dec2, _ = roc_curve(permuted_choices_dec2, test_output_dec_2)

                auc_boot_dec1[i] = auc(fpr_perm_dec1, tpr_perm_dec1)
                auc_boot_dec2[i] = auc(fpr_perm_dec2, tpr_perm_dec2)


            if sum(auc_neuron_dec1 > auc_boot_dec1)/n_boot > 0.975 and auc_neuron_dec1 > threshold:
                auc_neurons_mod[neuron, 2] = 1
            elif sum(auc_neuron_dec1 > auc_boot_dec1)/n_boot < 0.025 and auc_neuron_dec1 < (1-threshold):
                auc_neurons_mod[neuron, 2] = -1

            if sum(auc_neuron_dec2 > auc_boot_dec2)/n_boot > 0.975 and auc_neuron_dec2 > threshold:
                auc_neurons_mod[neuron, 3] = 1
            elif sum(auc_neuron_dec2 > auc_boot_dec2)/n_boot < 0.025 and auc_neuron_dec2 < (1-threshold):
                auc_neurons_mod[neuron, 3] = -1


        auc_neurons_choice = np.zeros((num_neurons, 2)) #1: auc, #2: -1, 0, 1

        for neuron in range(num_neurons):
            test_output = np.mean(rnn_output[correct_trials, -20:, neuron], axis=1)

            fpr, tpr, thresholds = roc_curve(choose_1, test_output)

            auc_neuron = auc(fpr,tpr)
            auc_neurons_choice[neuron, 0] = auc_neuron

            auc_boot = np.zeros(n_boot)

            for i in range(n_boot):
                permuted_choices = np.random.permutation(choose_1)
                fpr_perm, tpr_perm, _ = roc_curve(permuted_choices, test_output)
                auc_boot[i] = auc(fpr_perm, tpr_perm)


            if sum(auc_neuron > auc_boot)/n_boot > 0.975 and auc_neuron > threshold:
                auc_neurons_choice[neuron, 1] = 1
            elif sum(auc_neuron > auc_boot)/n_boot < 0.025 and auc_neuron < (1-threshold):
                auc_neurons_choice[neuron, 1] = -1

        print(f'third check')
        left_mod = np.nonzero(auc_neurons_mod[:, 2])[0]
        right_mod = np.nonzero(auc_neurons_mod[:, 3])[0]
        all_modality_neurons = np.union1d(left_mod, right_mod)

        choice_selective = np.nonzero(auc_neurons_choice[:, 1])[0]

        mixed_selective = np.intersect1d(all_modality_neurons, choice_selective)
        pure_choice_selective = list(set(choice_selective) - set(mixed_selective))
        pure_modality_selective = list(set(all_modality_neurons) - set(mixed_selective))

        selectivities = np.zeros(num_neurons)
        selectivities[pure_modality_selective] = 1
        selectivities[pure_choice_selective] = 2
        selectivities[mixed_selective] = 3

        #saving
        save_path = os.path.join(data_path, netw)
        np.save(os.path.join(save_path, 'unit_selectivity_roc_correctonly.npy'), selectivities)
        np.save(os.path.join(save_path, 'auc_neurons_mod_roc.npy'), auc_neurons_mod)
        np.save(os.path.join(save_path, 'auc_neurons_choice_roc.npy'), auc_neurons_choice)

        print(f'Done after {np.round((time.time() - start_time)/60, 2)} minutes')

if __name__ == "__main__":
    main()
