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
import matplotlib
import matplotlib.pyplot as plt
sys.path.append(os.path.expanduser('~/gym'))
sys.path.append(os.path.expanduser('~/stable-baselines'))
sys.path.append(os.path.expanduser('~/neurogym'))
import gym
import neurogym  # need to import it so ngym envs are registered


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


def get_dataset_for_SL(env_name, kwargs, rollout, n_tr=1000000,
                       nstps_test=1000, verbose=0,
                       seed=None):
    env = gym.make(env_name, **kwargs)
    env.seed(seed)
    env.reset()
    # TODO: this assumes 1-D observations
    samples = np.empty((TOT_TIMESTEPS, OBS_SIZE))
    target = np.empty((TOT_TIMESTEPS, ACT_SIZE))
    if verbose:
        num_steps_per_trial = int(nstps_test/env.num_tr)
        print('Task: ', env_name)
        print('Producing dataset with {0} steps'.format(TOT_TIMESTEPS) +
              ' and {0} trials'.format(n_tr) +
              ' ({0} steps per trial)'.format(num_steps_per_trial))
    count_stps = 0
    for tr in range(n_tr):
        obs = env.obs
        gt = env.gt
        samples[count_stps:count_stps+obs.shape[0], :] = obs
        target[count_stps:count_stps+gt.shape[0], :] = np.eye(ACT_SIZE)[gt]
        count_stps += obs.shape[0]
        assert obs.shape[0] == gt.shape[0]
        env.new_trial()

    samples = samples[:count_stps, :]
    target = target[:count_stps, :]
    samples = samples.reshape((-1, rollout, OBS_SIZE))
    target = target.reshape((-1, rollout, ACT_SIZE))
    return samples, target, env


def train_env_keras_net(env_name, kwargs, folder, rollout, num_h=256,
                        b_size=128, num_tr=200000, ntr_save=1000,
                        tr_per_ep=1000, verbose=1):
    env = test_env(env_name, kwargs=kwargs, num_steps=1)
    # from https://www.tensorflow.org/guide/keras/rnn
    xin = Input(batch_shape=(None, rollout, env.observation_space.shape[0]),
                dtype='float32')
    seq = LSTM(num_h, return_sequences=True)(xin)
    mlp = TimeDistributed(Dense(env.action_space.n, activation='softmax'))(seq)
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
                                                n_tr=tr_per_ep)
        model.fit(samples, target, epochs=1, verbose=0)
        # test
        samples, target, env = get_dataset_for_SL(env_name=env_name,
                                                  kwargs=kwargs,
                                                  rollout=rollout,
                                                  n_tr=tr_per_ep, seed=ind_ep)
        loss, acc = model.evaluate(samples, target, verbose=0)
        loss_training.append(loss)
        acc_training.append(acc)
        perf = eval_net_in_task(model, env_name=env_name, kwargs=kwargs,
                                tr_per_ep=ntr_save, rollout=rollout,
                                samples=samples, target=target, folder=folder,
                                show_fig=False, seed=ind_ep,
                                save=True, ntr_save=ntr_save)
        perf_training.append(perf)
        if verbose and ind_ep % 100 == 0:
            print('Accuracy: ', acc)
            print('Performance: ', perf)
            rem_time = (num_ep-ind_ep)*(time.time()-start_time)/3600
            print('epoch {0} out of {1}'.format(ind_ep, num_ep))
            print('remaining time: {:.2f}'.format(rem_time))
            print('-------------')

    data = {'acc': acc_training, 'loss': loss_training,
            'perf': perf_training}
    np.savez(folder + 'training.npz', **data)
    fig = plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(acc_training)
    plt.subplot(1, 3, 2)
    plt.plot(loss_training)
    plt.subplot(1, 3, 3)
    plt.plot(perf_training)

    fig.savefig(folder + 'performance.png')
    plt.close(fig)
    return model


def eval_net_in_task(model, env_name, kwargs, tr_per_ep, rollout, sl='SL',
                     samples=None, target=None, seed=0, show_fig=False,
                     folder='', save=False, ntr_save=1000):
    if samples is None:
        samples, target, _ = get_dataset_for_SL(env_name=env_name,
                                                kwargs=kwargs,
                                                rollout=rollout,
                                                n_tr=tr_per_ep, seed=seed)
    if sl == 'SL':
        actions = model.predict(samples)
    env = gym.make(env_name, **kwargs)
    env.seed(seed=seed)
    if save:
        env = manage_data.ManageData(env, folder=folder,
                                     num_tr_save=ntr_save)
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
        if sl == 'SL':
            action = actions[int(np.floor(index/rollout)),
                             (index % rollout), :]
            action = np.argmax(action)
        else:
            action, _ = model.predict([obs])
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
        n_stps_plt = 100
        observations = np.array(observations)
        f = plt.figure()
        plt.subplot(3, 1, 1)
        plt.imshow(observations[:n_stps_plt, :].T, aspect='auto')
        plt.title('observations')
        plt.subplot(3, 1, 2)
        plt.plot(np.arange(n_stps_plt)+1, actions_plt[:n_stps_plt], marker='+')
        gt = np.array(gt)
        if len(gt.shape) == 2:
            gt = np.argmax(gt, axis=1)
        plt.plot(np.arange(n_stps_plt)+1, gt[:n_stps_plt], 'r')
        # plt.plot(np.arange(n_stps_plt)+1, target_mat[:n_stps_plt], '--y')
        plt.title('actions')
        plt.xlim([-0.5, n_stps_plt+0.5])
        plt.subplot(3, 1, 3)
        plt.plot(np.arange(n_stps_plt)+1, rewards[:n_stps_plt], 'r')
        plt.title('reward')
        plt.xlim([-0.5, n_stps_plt+0.5])
        plt.title(str(np.mean(perf)))
        plt.tight_layout()
        plt.show()
        if folder != '':
            f.savefig(folder + 'task_struct.png')
            plt.close(f)

    return np.mean(perf)


if __name__ == '__main__':
    if len(sys.argv) != 6:
        raise ValueError('usage: bsls_run.py [model] [task]' +
                         '[seed] [num_trials] [rollout]')

    # ARGS
    alg = sys.argv[1]  # a2c acer acktr or ppo2
    task = sys.argv[2]  # ngym task (neurogym.all_tasks.keys())
    seed = int(sys.argv[3])
    num_trials = int(sys.argv[4])
    rollout = int(sys.argv[5])  # use 20 if short periods, else 100
    ntr_save = 1000
    dt = 100
    kwargs = {'dt': dt,
              'timing': dict(zip(states_list,
                                 zip(['constant']*len(states_list),
                                     [100]*len(states_list))))}

    # other relevant vars
    nstps_test = 1000
    env = test_env(task, kwargs=kwargs, num_steps=nstps_test)
    TOT_TIMESTEPS = int(nstps_test*num_trials/(env.num_tr))
    OBS_SIZE = env.observation_space.shape[0]
    ACT_SIZE = env.action_space.n

    savpath = os.path.expanduser(f'~/Jan2020/data/{alg}_{task}_{seed}.npz')
    main_folder = savpath[:-4] + '/'
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)

    model = train_env_keras_net(task, kwargs=kwargs, folder=main_folder,
                                rollout=rollout, num_tr=num_trials,
                                num_h=256, b_size=128,
                                tr_per_ep=1000, verbose=1)

    eval_net_in_task(model, task, kwargs=kwargs, tr_per_ep=1000,
                     rollout=rollout, show_fig=True, sl=alg,
                     folder=main_folder)
