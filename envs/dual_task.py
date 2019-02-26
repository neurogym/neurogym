#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 11:47:40 2018

@author: molano
"""
import sys
import itertools

import numpy as np
from gym import spaces

import ngym


class DualTask(ngym.ngym):  # TODO: task does not stop when breaking fixation
    def __init__(
            self,
            dt=0.2,
            exp_dur=1e5,
            block_dur=1e3,
            td=16.4,
            bt_tr_time=0.6,
            dpa_st=1.,
            dpa_d=1.,
            dpa_resp=0.4,
            do_gng_task=True,
            gng_time=4.,
            gng_st=0.6,
            gng_d=0.6,
            gng_resp=0.4,
            rewards=(0., -0.1, 1.0, -1.0),
            bg_noise=.01,
            perc_noise=0.1
    ):
        """The dual stimulus task.

        From https://www.biorxiv.org/content/10.1101/385393v1 by Cheng-Yu Li.

        Args:
            dt: float, dt
            exp_dur: float, experiment duration
            block_dur: float, block duration (s)
            td: float, ??
            bt_tr_time: float, ??
            dpa_st: float, ??
            dpa_resp: float, ??
            do_gng_task: bool, if True, do the Go/No-go task
            gng_time: float, ??
            gng_st: float, ??
            gng_d: float, ??
            gng_resp: float, ??
            rewads: set, ??
            bg_noise: float, background noise level
            perc_noise: float, perceptual noise level
        """
        # call the __init__ function from the super-class
        super().__init__(dt=dt)

        # experiment duration
        self.exp_dur = exp_dur

        # Actions are always 2: GO/NO-GO
        self.num_actions = 2

        # action space
        self.action_space = spaces.Discrete(self.num_actions)
        # observation space
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6, 1),
                                            dtype=np.float32)
        # rewards given for: fixating, stop fixating, correct, wrong
        self.rewards = rewards
        # stimuli identity: matching pairs are [0, 4] and [1, 5],
        # 2 and 3 are go and no-go respectivelly
        # TODO: revisit. Make eacch stimulus linear combination of all others?
        self.stims = np.array([0, 2, 1, 5, 3, 4]).reshape(3, 2)

        # independent background noise
        self.bg_noise = bg_noise

        # noise in perception (std of gaussians identifying each stimulus)
        self.perc_noise = perc_noise

        # boolean indicating whether the gng is included or only
        # the dpa tasks is done
        self.do_gng_task = do_gng_task
        if not self.do_gng_task:
            gng_time = 0

        # events times: when the go/no-go task is presented and the duration
        # of response window. The first stim. for DPA task is presented at 0
        # and the second at trial_duration
        eps = sys.float_info.epsilon
        self.gng_t = int((gng_time+eps)/dt)

        # duration (in trials) of blocks for the task rule
        self.block_dur = block_dur

        # stims duration
        self.gng_st = int((gng_st+eps)/dt)
        self.dpa_st = int((dpa_st+eps)/dt)

        # delays after gng and dpa tasks
        # TODO: make dpa2 time random?
        self.gng_d = int((gng_d+eps)/dt)
        self.dpa_d = int((dpa_d+eps)/dt)

        # response time
        self.gng_resp = int((gng_resp+eps)/dt)
        self.dpa_resp = int((dpa_resp+eps)/dt)

        # between trials time
        self.bt_tr_time = int((bt_tr_time+eps)/dt)

        # rule: first element corresponds to which element will be
        # associated with GO in the go/no-go task
        # second element corresponds to the two possible
        # couples (identified with 0 or 1) in the matching task
        self.rule_block_counter = 0
        self.rules = [p for p in itertools.product([0, 1], repeat=2)]
        if not self.do_gng_task:
            self.rules = self.rules[0:2]

        self.current_rule = self.rules[self.rule_block_counter]

    def step(self, action, net_state=[]):
        """
        receives an action and returns a new observation, a reward, a flag
        variable indicating whether to reset and some relevant info
        """
        # this is whether the RNN does well the dpa task.
        correct_dpa = False
        correct_gng = False
        reward = 0
        # checking periods
        gng, dpa, end_gng_flg, end_dpa_flg =\
            response_periods(self.t_stp, self.gng_t, self.gng_st,
                             self.gng_d, self.gng_resp, self.dpa_st,
                             self.dpa_d, self.dpa_resp,
                             self.bt_tr_time, self.td)
        # check go/no-go task
        if gng and self.do_gng_task and self.gng_flag:
            correct_gng, self.gng_flag =\
                rew_deliv_control(self.true[0], action)
            self.correct_gng = correct_gng
        # check dpa task
        elif dpa and self.dpa_flag:
            correct_dpa, self.dpa_flag =\
                rew_deliv_control(self.true[1], action)
            self.correct_dpa = correct_dpa
        elif end_gng_flg:
            reward = self.rewards[2+1*(not self.correct_gng)]
        elif end_dpa_flg:
            reward = self.rewards[2+1*(not self.correct_dpa)]
        else:
            if action == 0:
                reward = self.rewards[0]
            else:
                reward = self.rewards[1]

        new_trial = self.td == self.t_stp
        if new_trial:
            new_state = self._new_trial()
            # check if it is time to update the network
            done = self.num_tr >= self.exp_dur
        else:
            new_state = self._get_state()

        info = {}  # TODO
        return new_state, reward, done, info

    def _get_state(self):
        """Returns the corresponding state.

        Needs to check the go/no-go events to know whether to present a
        stimulus or just noise
        """
        self.t_stp += 1
        stim = -1
        # decide which stimulus to present
        dpa_1, gng, dpa_2 = stim_periods(self.t_stp, self.gng_t,
                                         self.gng_st, self.dpa_st,
                                         self.dpa_d, self.dpa_resp,
                                         self.bt_tr_time, self.td)
        if dpa_1:
            stim = self.stims[0, self.internal_state[0]]
        elif gng and self.do_gng_task:
            stim = self.stims[1, self.internal_state[1]]
        elif dpa_2:
            stim = self.stims[2, self.internal_state[2]]

        # if there is no stimulus, present bg noise
        if stim == -1:
            aux = self.rng.uniform(low=0., high=self.bg_noise, size=(6,))
        else:
            aux = self.rng.uniform(low=0., high=self.bg_noise, size=(6,))
            aux[stim] += 1.

        self.state = aux
        # reshape
        self.state = np.reshape(self.state, [1, np.size(self.stims), 1])

        return self.state

    def _new_trial(self):
        """
        new trial: reset the timesteps and increment the number of trials
        """
        self.num_tr += 1
        self.t_stp = 0
        # this is whether the RNN does well the go/no-go task.
        # I put it here because it needs to be stored until the trial ends
        self.correct_gng = False
        self.correct_dpa = False

        # flag to control that the reward is not given twice
        self.gng_flag = True
        self.dpa_flag = True

        # choose a stimulus for each event: 1st stim for DPA task,
        # stim for gng task, 2nd stim for DPA task
        self.internal_state = self.rng.choice([0, 1], (3, 1))

        # decide the position of the stims
        # if the block is finished change the rule
        if self.num_tr % self.block_dur == 0:
            self.rule_block_counter += 1
            self.rule_block_counter = self.rule_block_counter % len(self.rules)
            self.current_rule = self.rules[self.rule_block_counter]

        # the correct actions
        match = np.abs(self.internal_state[0]-self.internal_state[2])
        self.true = [self.internal_state[1] == self.current_rule[0],
                     match == self.current_rule[1]]

        # store some data about trial
        int_st_aux = self.internal_state.copy()
        self.stms_conf.append(int_st_aux)
        self.rule_mat.append(self.current_rule)

        # get state
        s = self.get_state()

        # during some episodes I save all data points
        aux = np.floor(self.num_tr / self.num_tr_svd)
        if aux % self.sv_pts_stp == 0:
            self.all_pts_data.update(new_state=s, new_trial=1,
                                     num_trials=self.num_tr,
                                     stim_conf=int_st_aux)

        return s

# STIMULUS AND REWARD PERIODS


def rew_deliv_control(gt, action):
    correct = gt[0] == action  # correct when action == ground truth
    flag = action == 0  # the flag is only false when the net goes
    return correct, flag


def stim_periods(t_stp,
                 gng_t, gng_st, dpa_st, dpa_d, dpa_resp,
                 bt_tr_time, td):
    """Get the stimulus periods."""
    # first dpa stim period
    dpa_1 = 1 <= t_stp <= dpa_st
    # gng stim period
    gng = gng_t < t_stp <= gng_t+gng_st
    # second dpa stim period
    start_second_stim, end_second_stim =\
        dpa2_stim(dpa_st, dpa_d, dpa_resp, bt_tr_time, td)
    dpa_2 = start_second_stim < t_stp <= end_second_stim

    return dpa_1, gng, dpa_2


def response_periods(t_stp, gng_t, gng_st, gng_d, gng_resp,
                     dpa_st, dpa_d, dpa_resp, bt_tr_time, td):
    """Get the response periods.

    Returns:
        gng: bool, True if t_stp is in Go/No-go period
        dpa: bool, True if in DPA period
        end_gng_flg: ??
        end_dpa_flg: ??
    """
    start_gng_resp, end_gng_resp = gng_r(gng_t, gng_st, gng_d, gng_resp)
    gng = start_gng_resp < t_stp <= end_gng_resp
    #
    start_dpa_resp, end_dpa_resp = dpa2_r(dpa_resp, bt_tr_time, td)
    dpa = start_dpa_resp < t_stp <= end_dpa_resp

    end_gng_flg = t_stp == end_gng_resp + 1
    end_dpa_flg = t_stp == end_dpa_resp + 1
    return gng, dpa, end_gng_flg, end_dpa_flg


def dpa2_stim(dpa_st, dpa_d, dpa_resp, bt_tr_time, td):
    """DPA stimulus period."""
    last_periods = dpa_st+dpa_d+dpa_resp+bt_tr_time
    dpa_2_st = td-last_periods
    dpa_2_end = td-last_periods + dpa_st
    return dpa_2_st, dpa_2_end


def dpa2_r(dpa_resp, bt_tr_time, td):
    """DPA response period."""
    last_periods = dpa_resp+bt_tr_time
    start_dpa_resp = td-last_periods
    end_dpa_resp = td-last_periods + dpa_resp
    return start_dpa_resp, end_dpa_resp


def gng_r(gng_t, gng_st, gng_d, gng_resp):
    """Go/No-go response period."""
    start_gng_resp = gng_t+gng_st+gng_d
    end_gng_resp = gng_t+gng_st+gng_d+gng_resp
    return start_gng_resp, end_gng_resp
