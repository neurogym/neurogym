#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test training according to params (model, task, seed)
"""
import os
import sys
import numpy as np
import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, Input
import matplotlib.pyplot as plt
import gym
import neurogym  # need to import it so ngym envs are registered
from neurogym.meta import tasks_info


def test_env(env, kwargs, num_steps=100):
    """Test if all one environment can at least be run."""
    env = gym.make(env, **kwargs)
    env.reset()
    for stp in range(num_steps):
        action = env.action_space.sample()
        state, rew, done, info = env.step(action)
        if done:
            env.reset()
    return env


def get_dataset_for_SL(env_name, kwargs, rollout, n_tr, n_steps, obs_size,
                       act_size, nstps_test=1000, verbose=0,
                       seed=None):
    env = gym.make(env_name, **kwargs)
    env.seed(seed)
    env.reset()
    # TODO: this assumes 1-D observations
    samples = np.empty((n_steps, obs_size))
    target = np.empty((n_steps, act_size))
    if verbose:
        num_steps_per_trial = int(nstps_test/env.num_tr)
        print('Task: ', env_name)
        print('Producing dataset with {0} steps'.format(n_steps) +
              ' and {0} trials'.format(n_tr) +
              ' ({0} steps per trial)'.format(num_steps_per_trial))
    count_stps = 0
    for tr in range(n_tr):
        obs = env.obs
        gt = env.gt
        samples[count_stps:count_stps+obs.shape[0], :] = obs
        target[count_stps:count_stps+gt.shape[0], :] = np.eye(act_size)[gt]
        count_stps += obs.shape[0]
        assert obs.shape[0] == gt.shape[0]
        env.new_trial()

    samples = samples[:count_stps, :]
    target = target[:count_stps, :]
    samples = samples.reshape((-1, rollout, obs_size))
    target = target.reshape((-1, rollout, act_size))
    return samples, target, env


def train_env_keras_net(env_name, kwargs, rollout, num_tr, folder='',
                        num_h=256, b_size=128, ntr_save=1000,
                        tr_per_ep=1000, verbose=1, log_int=10):
    # get mean number of steps per trial to compute total number of steps
    nstps_test = 1000
    env = test_env(env_name, kwargs=kwargs, num_steps=nstps_test)
    num_stps_per_trial = nstps_test*num_tr/(env.num_tr)
    steps_per_tr = int(tr_per_ep*num_stps_per_trial)
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.n

    # from https://www.tensorflow.org/guide/keras/rnn
    xin = Input(batch_shape=(None, rollout, obs_size),
                dtype='float32')
    seq = LSTM(num_h, return_sequences=True)(xin)
    mlp = TimeDistributed(Dense(act_size, activation='softmax'))(seq)
    model = Model(inputs=xin, outputs=mlp)
    model.summary()
    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    num_ep = int(num_tr/tr_per_ep)
    loss_training = []
    acc_training = []
    perf_training = []
    for ind_ep in range(num_ep):
        start_time = time.time()
        # train
        samples, target, _ = get_dataset_for_SL(env_name=env_name,
                                                kwargs=kwargs,
                                                rollout=rollout,
                                                n_tr=tr_per_ep,
                                                n_steps=steps_per_tr,
                                                obs_size=obs_size,
                                                act_size=act_size)
        model.fit(samples, target, epochs=1, verbose=0)
        # test
        samples, target, env = get_dataset_for_SL(env_name=env_name,
                                                  kwargs=kwargs,
                                                  rollout=rollout,
                                                  n_tr=tr_per_ep,
                                                  n_steps=steps_per_tr,
                                                  obs_size=obs_size,
                                                  act_size=act_size,
                                                  seed=ind_ep)
        loss, acc = model.evaluate(samples, target, verbose=0)
        loss_training.append(loss)
        acc_training.append(acc)
        perf = eval_net_in_task(model, env_name=env_name, kwargs=kwargs,
                                tr_per_ep=ntr_save, rollout=rollout,
                                samples=samples, target=target, folder=folder,
                                show_fig=(ind_ep == (num_ep-1)), seed=ind_ep)
        perf_training.append(perf)
        if verbose and ind_ep % log_int == 0:
            print('Accuracy: ', acc)
            print('Performance: ', perf)
            rem_time = (num_ep-ind_ep)*(time.time()-start_time)/3600
            print('epoch {0} out of {1}'.format(ind_ep, num_ep))
            print('remaining time: {:.2f}'.format(rem_time))
            print('-------------')

    data = {'acc': acc_training, 'loss': loss_training,
            'perf': perf_training}

    fig = plt.figure(figsize=(8,8))
    plt.subplot(1, 3, 1)
    plt.plot(np.arange(len(acc_training))*tr_per_ep, acc_training)
    plt.title('Accuracy')
    plt.xlabel('Trials')
    plt.subplot(1, 3, 2)
    plt.plot(np.arange(len(acc_training))*tr_per_ep, loss_training)
    plt.title('Loss')
    plt.xlabel('Trials')
    plt.subplot(1, 3, 3)
    plt.plot(np.arange(len(acc_training))*tr_per_ep, perf_training)
    plt.title('performance (accuracy decision period)')
    plt.xlabel('Trials')
    if folder != '':
        np.savez(folder + 'training.npz', **data)
        fig.savefig(folder + 'performance.png')
        plt.close(fig)
    return model


def eval_net_in_task(model, env_name, kwargs, tr_per_ep, rollout,
                     samples, target, seed, show_fig=False,
                     folder=''):

    actions = model.predict(samples)
    env = gym.make(env_name, **kwargs)
    env.seed(seed=seed)
    obs = env.reset()
    perf = []
    actions_plt = []
    rew_temp = []
    observations = []
    rewards = []
    gt = []
    target_mat = []
    action = 0
    num_steps = int(samples.shape[0]*samples.shape[1])
    for ind_act in range(num_steps-1):
        index = ind_act + 1
        observations.append(obs)
        action = actions[int(np.floor(index/rollout)),
                         (index % rollout), :]
        action = np.argmax(action)
        obs, rew, _, info = env.step(action)
        if info['new_trial']:
            perf.append(rew)
        if show_fig:
            rew_temp.append(rew)
            rewards.append(rew)
            gt.append(info['gt'])
            target_mat.append(target[int(np.floor(index/rollout)),
                                     index % rollout])
            actions_plt.append(action)

    if show_fig:
        observations = np.array(observations)
        f = tasks_info.fig_(obs=observations, actions=actions_plt, gt=gt,
                            rewards=rewards, n_stps_plt=100, perf=perf,
                            legend=True, name='')
        if folder != '':
            f.savefig(folder + 'task_struct.png')
            plt.close(f)

    return np.mean(perf)


if __name__ == '__main__':
    # ARGS
    task = 'RDM-v0'
    num_trials = 100000
    rollout = 20
    dt = 100
    kwargs = {'dt': 100, 'timing': {'fixation': ('constant', 200),
                                    'stimulus': ('constant', 200),
                                    'decision': ('constant', 100)}}
    model = train_env_keras_net(task, kwargs=kwargs,
                                rollout=rollout, num_tr=num_trials,
                                num_h=256, b_size=128,
                                tr_per_ep=1000, verbose=1)
