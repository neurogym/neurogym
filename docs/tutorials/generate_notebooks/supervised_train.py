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

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_modelpath(envid):
    # Make a local file directories
    path = Path('.') / 'files'
    os.makedirs(path, exist_ok=True)
    path = path / envid
    os.makedirs(path, exist_ok=True)
    return path


# Define network
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, hidden = self.lstm(x)
        x = self.linear(out)
        return x, out


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
        'batch_size': 16,
        'seq_len': 100,
        'envid': envid,
    }

    env_kwargs = {'dt': config['dt']}
    config['env_kwargs'] = env_kwargs

    # Save config
    with open(modelpath / 'config.json', 'w') as f:
        json.dump(config, f)

    # Make supervised dataset
    dataset = ngym.Dataset(
        envid, env_kwargs=env_kwargs, batch_size=config['batch_size'],
        seq_len=config['seq_len'])
    env = dataset.env
    act_size = env.action_space.n
    # Train network
    net = Net(input_size=env.observation_space.shape[0],
              hidden_size=config['hidden_size'],
              output_size=act_size)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'])

    print('Training task ', envid)

    running_loss = 0.0
    for i in range(2000):
        inputs, labels = dataset()
        inputs = torch.from_numpy(inputs).type(torch.float).to(device)
        labels = torch.from_numpy(labels.flatten()).type(torch.long).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, _ = net(inputs)

        loss = criterion(outputs.view(-1, act_size), labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:
            print('{:d} loss: {:0.5f}'.format(i + 1, running_loss / 200))
            running_loss = 0.0
            torch.save(net.state_dict(), modelpath / 'net.pth')

    print('Finished Training')


# # TODO: Make this into a function in neurogym
# perf = 0
# num_trial = 200
# for i in range(num_trial):
#     env.new_trial()
#     ob, gt = env.ob, env.gt
#     ob = ob[:, np.newaxis, :]  # Add batch axis
#     inputs = torch.from_numpy(ob).type(torch.float).to(device)
#
#     action_pred = net(inputs)
#     action_pred = action_pred.detach().numpy()
#     action_pred = np.argmax(action_pred, axis=-1)
#     perf += gt[-1] == action_pred[-1, 0]
#
# perf /= num_trial
# print('Average performance in {:d} trials'.format(num_trial))
# print(perf)


def infer_test_timing(env):
    """Infer timing of environment for testing."""
    timing = {}
    for period in env.timing.keys():
        period_times = [env.sample_time(period) for _ in range(100)]
        timing[period] = np.median(period_times)
    return timing


def run_network(envid):
    """Run trained networks for analysis.

    Args:
        envid: str, Environment ID

    Returns:
        activity: a list of activity matrices, each matrix has shape (
        N_time, N_neuron)
        info: pandas dataframe, each row is information of a trial
        config: dict of network, training configurations
    """
    modelpath = get_modelpath(envid)
    with open(modelpath / 'config.json') as f:
        config = json.load(f)

    env_kwargs = config['env_kwargs']

    # Run network to get activity and info
    # Environment
    env = gym.make(envid, **env_kwargs)
    env.timing = infer_test_timing(env)
    env.reset(no_step=True)

    # Instantiate the network and print information
    with torch.no_grad():
        net = Net(input_size=env.observation_space.shape[0],
                  hidden_size=config['hidden_size'],
                  output_size=env.action_space.n)
        net = net.to(device)
        net.load_state_dict(torch.load(modelpath / 'net.pth'))

        perf = 0
        num_trial = 100

        activity = list()
        info = pd.DataFrame()

        for i in range(num_trial):
            env.new_trial()
            ob, gt = env.ob, env.gt
            inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float)
            action_pred, hidden = net(inputs)

            # Compute performance
            action_pred = action_pred.detach().numpy()
            choice = np.argmax(action_pred[-1, 0, :])
            correct = choice == gt[-1]

            # Log trial info
            trial_info = env.trial
            trial_info.update({'correct': correct, 'choice': choice})
            info = info.append(trial_info, ignore_index=True)

            # Log stimulus period activity
            activity.append(np.array(hidden)[:, 0, :])

        print('Average performance', np.mean(info['correct']))

    activity = np.array(activity)
    return activity, info, config
