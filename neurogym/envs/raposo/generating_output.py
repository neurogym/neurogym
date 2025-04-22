import os
import pickle
import time
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from network import CustomRNN
from raposo_task import RaposoTask

#! This fixes possible errors when running on M1/M2 but could have side effects:
#! https://github.com/dmlc/xgboost/issues/1715
os.environ['KMP_DUPLICATE_LIB_OK']='True'

n_networks = 100 #number of networks
first_network = 38 # number for the first network

save_path = '/Users/lexotto/Documents_Mac/Stage/UVA/Code/BehavioralVariabilityRNN-main/data/good_runs'
n_rnn_output_split = 2

# instantiate task class
task_param = {}
tau = 100 #time constant

task_param['fmin'] = 9
task_param['fmax'] = 16
task_param['freq_step'] = 1  # difference between two consecutive frequencies
task_param['std_inp_noise'] = 0.01
task_param['fixation'] = True
task_param['tau'] = tau
task = RaposoTask(task_param)

# RNN settings
rnn_param = {}
rnn_param['hidden_size'] = 150
rnn_param['ex_in_ratio'] = 0.8
rnn_param['rec_noise_std'] = 0.15

# training hyperparameters
test_size = 8192
test_minibatch_size = 4
lr = 0.01
dt_test = 2


# begin simulationsn

for n in tqdm(range(first_network, first_network + n_networks)):
    start_time = time.time()
    print(f'Started {n}')
    rng = np.random.default_rng()  # seed for generating trials

    # create the network
    net = CustomRNN(task.Nin, rnn_param['hidden_size'], task.Nout, rnn_param)
    # save the created network
    if not os.path.exists(os.path.join(save_path, str(n))):
    #     os.makedirs(os.path.join(save_path, str(n)))
    # torch.save(net.state_dict(), os.path.join(save_path, str(n), 'model'))
        print(f'Network {n} not in good_runs, and therefore skipped.')
        continue
    if os.path.exists(os.path.join(save_path, str(n),'model')):
            net.load_state_dict(torch.load(os.path.join(save_path, str(n), 'model')))
            print(f'Model {n} loaded')
    else:
        print(f'Model {n} not found')
        continue

    with open(os.path.join(save_path, str(n), 'accuracy.txt'), 'a+') as file_accuracy:
        losses = []
        accuracies = []
        keep_running = True
        num_epochs = 1000
        validation_percentages_analyzed = []
        validation_accuracies = []

        test_trials = task.generate_trials(rng, dt_test, test_size)
        test_output = np.zeros((test_size, test_trials['inputs'].shape[1] + 1, 2))
        rnn_output = np.zeros((test_size, test_trials['inputs'].shape[1] + 1, rnn_param['hidden_size']))
        n_test_batch = int(test_size / test_minibatch_size)

        DECISION_THRESHOLD = 0.2

        n_test_correct = 0

        for j in range(n_test_batch):
            start_idx = j * test_minibatch_size
            end_idx = (j + 1) * test_minibatch_size

            cur_batch = test_trials['inputs'][start_idx:end_idx]
            cur_batch_choice = test_trials['choice'][start_idx:end_idx]
            cur_batch_phases = test_trials['phases']


            test_batch_output, rnn_batch_output = net(torch.Tensor(cur_batch), tau, dt_test)

            rnn_output[start_idx:end_idx] = rnn_batch_output.detach().numpy()
            test_output[start_idx:end_idx] = test_batch_output.detach().numpy()

            output = test_batch_output.detach().numpy()
            out_diff = output[:, cur_batch_phases['stimulus'], 1] - output[:, cur_batch_phases['stimulus'], 0]

            decision_time = np.argmax(np.abs(out_diff) > DECISION_THRESHOLD, axis=1)

            analysed_trials = np.nonzero(decision_time != 0)[0]

            prediction = (out_diff[analysed_trials, decision_time[analysed_trials]] > 0).astype(np.int_)

            n_test_correct += np.sum(prediction == cur_batch_choice[analysed_trials])
            if j % 100 == 0:
                print('Processed batch ' + str(j) + '/' + str(n_test_batch))

        test_accuracy = n_test_correct * 100 / test_size
        print('Testing accuracy:' + str(test_accuracy))

        np.save(os.path.join(save_path, str(n), f'test_output.npy'), test_output)
        with open(os.path.join(save_path, str(n), f'test_trials.pkl'), 'wb') as f:
            pickle.dump(test_trials, f, pickle.HIGHEST_PROTOCOL)

        sample_per_file = test_size // n_rnn_output_split
                # save rnn_output
        if not os.path.exists(os.path.join(save_path, str(n), 'batch2')):
            os.makedirs(os.path.join(save_path, str(n), 'batch2'))
        for s in range(n_rnn_output_split):
            file_sample = rnn_output[s * sample_per_file:(s + 1) * sample_per_file, :, :]
            np.save(os.path.join(save_path, str(n), 'batch2', f'rnn_output_p{s + 1}.npy'), file_sample)
