#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:45:33 2019

@author: molano
"""


import matplotlib.pyplot as plt
import numpy as np

import sys
import gym
from os.path import expanduser
from gym import spaces
import neurogym as ngym
home = expanduser("~")
sys.path.append(home)
sys.path.append(home + '/neurogym')
sys.path.append(home + '/gym')


class CVLearning(ngym.EpochEnv):
    metadata = {
        'description': 'Implements shaping for the delay-response task,' +
        ' in which agents have to integrate two stimuli and report' +
        ' which one is larger on average after a delay.',
        'paper_link': 'https://www.nature.com/articles/s41586-019-0919-7',
        'paper_name': 'Discrete attractor dynamics underlies persistent' +
        ' activity in the frontal cortex',
        'timing': {
            'fixation': ('constant', 200),
            'stimulus': ('constant', 1150),
            'delay': ('choice', [300, 500, 700, 900, 1200, 2000, 3200, 4000]),
            # 'go_cue': ('constant', 100), # TODO: Not implemented
            'decision': ('constant', 1500)},
        'stimEv': 'Controls the difficulty of the experiment. (def: 1.)',
    }

    def __init__(self, dt=100, timing=None, stimEv=1., perf_w=1000,
                 max_num_reps=3, init_ph=0, th=0.8):
        super().__init__(dt=dt, timing=timing)
        self.choices = [1, 2]
        # cohs specifies the amount of evidence (which is modulated by stimEv)
        self.cohs = np.array([0, 6.4, 12.8, 25.6, 51.2])*stimEv
        # Input noise
        sigma = np.sqrt(2*100*0.01)
        self.sigma_dt = sigma / np.sqrt(self.dt)
        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_FAIL = -1.
        self.R_MISS = 0.
        self.abort = False
        self.firstcounts = True
        self.first_flag = False
        self.curr_ph = init_ph
        self.curr_perf = 0
        self.perf_window = perf_w
        self.goal_perf = [th]*4
        self.mov_window = []
        self.counter = 0
        self.max_num_reps = max_num_reps
        self.rew = 0
        # action and observation spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)

    def new_trial(self, **kwargs):
        """
        new_trial() is called when a trial ends to generate the next trial.
        The following variables are created:
            durations, which stores the duration of the different periods (in
            the case of rdm: fixation, stimulus and decision periods)
            ground truth: correct response for the trial
            coh: stimulus coherence (evidence) for the trial
            obs: observation
        """
        self.trial = {
            'ground_truth': self.rng.choice(self.choices),
            'coh': self.rng.choice(self.cohs),
            'sigma_dt': self.sigma_dt,
        }

        # init durations with None
        self.durs = {key: None for key in self.metadata['timing']}

        self.first_choice_rew = None
        # self.set_phase()
        if self.curr_ph == 0:
            # no stim, reward is in both left and right
            # agent cannot go N times in a row to the same side
            if np.abs(self.counter) >= self.max_num_reps:
                ground_truth = 1 if self.action == 2 else 2
                self.trial.update({'ground_truth': ground_truth})
                self.R_FAIL = 0
            else:
                self.R_FAIL = self.R_CORRECT
            self.durs.update({'stimulus': (0),
                             'delay': (0)})
            self.trial.update({'sigma_dt': 0})

        elif self.curr_ph == 1:
            # stim introduced with no ambiguity
            # wrong answer is not penalized
            # agent can keep exploring until finding the right answer
            self.durs.update({'delay': (0)})
            self.trial.update({'coh': 100})
            self.trial.update({'sigma_dt': 0})
            self.R_FAIL = 0
            self.firstcounts = False
        elif self.curr_ph == 2:
            # first answer counts
            # wrong answer is penalized
            self.durs.update({'delay': (0)})
            self.trial.update({'coh': 100})
            self.trial.update({'sigma_dt': 0})
            self.R_FAIL = -1
            self.firstcounts = True
        elif self.curr_ph == 3:
            # delay component is introduced
            self.trial.update({'coh': 100})
            self.trial.update({'sigma_dt': 0})
        # phase 4: ambiguity component is introduced

        self.first_flag = False

        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------

        self.trial.update(kwargs)

        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        self.add_epoch('fixation', after=0)
        self.add_epoch('stimulus', duration=self.durs['stimulus'],
                       after='fixation')
        self.add_epoch('delay', duration=self.durs['delay'],
                       after='stimulus')
        self.add_epoch('decision', after='delay', last_epoch=True)

        # define observations
        self.set_ob('fixation', [1, 0, 0])
        stim = self.view_ob('stimulus')
        stim[:, 0] = 1
        stim[:, 1:] = (1 - self.trial['coh']/100)/2
        stim[:, self.trial['ground_truth']] = (1 + self.trial['coh']/100)/2
        stim[:, 1:] +=\
            np.random.randn(stim.shape[0], 2) * self.trial['sigma_dt']

        self.set_ob('delay', [1, 0, 0])

        self.set_groundtruth('decision', self.trial['ground_truth'])

    def count(self, action):
        '''
        check the last three answers during stage 0 so the network has to
        alternate between left and right
        '''
        if action != 0:
            new = action - 2/action
            if np.sign(self.counter) == np.sign(new):
                self.counter += new
            else:
                self.counter = new

    def set_phase(self):
        if self.curr_ph < 4:
            if len(self.mov_window) >= self.perf_window:
                self.mov_window.append(1*(self.rew == self.R_CORRECT))
                self.mov_window.pop(0)  # remove first value
                self.curr_perf = np.sum(self.mov_window)/self.perf_window
                if self.curr_perf >= self.goal_perf[self.curr_ph]:
                    self.curr_ph += 1
                    self.mov_window = []
            else:
                self.mov_window.append(1*(self.rew == self.R_CORRECT))

    def _step(self, action):
        # obs, reward, done, info = self.env._step(action)
        # ---------------------------------------------------------------------
        # Reward and observations
        # ---------------------------------------------------------------------
        new_trial = False
        # rewards
        reward = 0
        # observations
        gt = self.gt_now

        first_choice = False
        if self.in_epoch('fixation') or self.in_epoch('delay'):
            if action != 0:
                new_trial = self.abort
                reward = self.R_ABORTED
        elif self.in_epoch('decision'):
            if action == gt:
                reward = self.R_CORRECT
                new_trial = True
                if ~self.first_flag:
                    first_choice = True
                    self.first_flag = True
            elif action == 3 - gt:  # 3-action is the other act
                reward = self.R_FAIL
                new_trial = self.firstcounts
                if ~self.first_flag:
                    first_choice = True
                    self.first_flag = True

        info = {'new_trial': new_trial, 'gt': gt,
                'curr_ph': self.curr_ph, 'first_rew': self.rew}

        # check if first choice (phase 1)
        if ~self.firstcounts and first_choice:
            self.first_choice_rew = reward

        # set reward for all phases
        self.rew = self.first_choice_rew or reward

        if info['new_trial']:
            self.set_phase()
            if self.curr_ph == 0:
                # control that agent does not repeat side more than 3 times
                self.count(action)
                self.action = action

        return self.obs_now, reward, False, info


def plot_struct(env, num_steps_env=200, n_stps_plt=200,
                def_act=None, model=None, name=None, legend=True):
    if isinstance(env, str):
        env = gym.make(env)
    if name is None:
        name = type(env).__name__
    observations = []
    obs_cum = []
    state_mat = []
    rewards = []
    actions = []
    actions_end_of_trial = []
    gt = []
    perf = []

    obs = env.reset()
    obs_cum_temp = obs
    for stp in range(int(num_steps_env)):
        if model is not None:
            action, _states = model.predict(obs)
            action = [action]
            state_mat.append(_states)
        elif def_act is not None:
            action = def_act
        else:
            action = env.gt_now
        obs, rew, done, info = env.step(action)
        obs_cum_temp += obs
        obs_cum.append(obs_cum_temp.copy())
        if isinstance(info, list):
            info = info[0]
            obs_aux = obs[0]
            rew = rew[0]
            done = done[0]
            action = action[0]
            rew_correct = env.get_attr('R_CORRECT')
        else:
            obs_aux = obs
            rew_correct = env.R_CORRECT

        if done:
            env.reset()
        observations.append(obs_aux)
        if info['new_trial']:
            actions_end_of_trial.append(action)
            perf.append(rew == rew_correct)
            obs_cum_temp = np.zeros_like(obs_cum_temp)
        else:
            actions_end_of_trial.append(-1)
        rewards.append(rew)
        actions.append(action)
        gt.append(info['gt'])
    if model is not None:
        states = np.array(state_mat)
        states = states[:, 0, :]
    else:
        states = None
    obs_cum = np.array(obs_cum)
    obs = np.array(observations)
    fig_(obs, actions, gt, rewards, n_stps_plt, perf, legend=legend,
         model=model, states=states, name=name)
    data = {'obs': obs, 'obs_cum': obs_cum, 'rewards': rewards,
            'actions': actions, 'perf': perf,
            'actions_end_of_trial': actions_end_of_trial, 'gt': gt,
            'states': states}
    return data


def fig_(obs, actions, gt, rewards, n_stps_plt, perf, legend=True,
         obs_cum=None, model=None, states=None, name=''):
    if model is not None:
        rows = 4
    else:
        rows = 3

    f = plt.figure(figsize=(8, 8))
    # obs
    plt.subplot(rows, 1, 1)
    plt.imshow(obs[:n_stps_plt, :].T, aspect='auto')
    plt.title('observations ' + name + ' task')
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    # actions
    plt.subplot(rows, 1, 2)
    plt.plot(np.arange(n_stps_plt) + 0.,
             actions[:n_stps_plt], marker='+', label='actions')
    gt = np.array(gt)
    if len(gt.shape) == 2:
        gt = np.argmax(gt, axis=1)
    plt.plot(np.arange(n_stps_plt) + 0.,
             gt[:n_stps_plt], 'r', label='ground truth')
    plt.ylabel('actions')
    if legend:
        plt.legend()
    plt.xlim([-0.5, n_stps_plt-0.5])
    ax = plt.gca()
    ax.set_xticks([])
    # rewards
    plt.subplot(rows, 1, 3)
    plt.plot(np.arange(n_stps_plt) + 0.,
             rewards[:n_stps_plt], 'r')
    plt.xlim([-0.5, n_stps_plt-0.5])
    plt.ylabel('reward ' + ' (' + str(np.round(np.mean(perf), 2)) + ')')
    if model is not None:
        ax = plt.gca()
        ax.set_xticks([])
        plt.subplot(rows, 1, 4)
        plt.imshow(states[:n_stps_plt, int(states.shape[1]/2):].T,
                   aspect='auto')
        plt.title('network activity')
        plt.ylabel('neurons')
        ax = plt.gca()

    plt.xlabel('timesteps')
    plt.tight_layout()
    plt.show()
    return f


if __name__ == '__main__':
    env = CVLearning(init_ph=0)
    plot_struct(env, num_steps_env=2000, n_stps_plt=2000)
