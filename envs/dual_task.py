#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 11:47:40 2018

@author: molano
"""
import numpy as np
import itertools
import tasks
import data
import sys
import matplotlib.pyplot as plt
import matplotlib
# parameters for figures
left = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.4   # the amount of width reserved for blank space between subplots
hspace = 0.4   # the amount of height reserved for white space between subplots
line_width = 2


class dual_task(tasks.task):
    def __init__(self, update_net_step=100,
                 Dt=0.2, td=16.4, gng_time=4., bt_tr_time=0.6,
                 dpa_st=1., dpa_d=1., dpa_resp=0.4,
                 gng_st=0.6, gng_d=0.6, gng_resp=0.4,
                 rewards=(-0.1, 0.0, 1.0, -1.0),
                 bg_noise=.01, perc_noise=0.1,
                 folder='', block_dur=200, do_gng_task=True):
        # call the __init__ function from the super-class
        super().__init__(trial_duration=int(td/Dt),
                         update_net_step=update_net_step,
                         rewards=rewards, folder=folder)

        # Actions are always 2: GO/NO-GO
        self.num_actions = 2

        # stimuli identity: matching pairs are [0, 4] and [1, 5],
        # 2 and 3 are go and no-go respectivelly
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
        self.gng_t = int((gng_time+eps)/Dt)

        # duration (in trials) of blocks for the task rule
        self.block_dur = block_dur

        # stims duration
        self.gng_st = int((gng_st+eps)/Dt)
        self.dpa_st = int((dpa_st+eps)/Dt)

        # delays after gng and dpa tasks
        self.gng_d = int((gng_d+eps)/Dt)
        self.dpa_d = int((dpa_d+eps)/Dt)

        # response time
        self.gng_resp = int((gng_resp+eps)/Dt)
        self.dpa_resp = int((dpa_resp+eps)/Dt)

        # between trials time
        self.bt_tr_time = int((bt_tr_time+eps)/Dt)

        # time corresponding to a single step (s)
        self.Dt = Dt

        # rule: first element corresponds to which element will be
        # associated with GO in the go/no-go task
        # second element corresponds to the two possible
        # couples (identified with 0 or 1) in the matching task
        self.rule_block_counter = 0
        self.rules = [p for p in itertools.product([0, 1], repeat=2)]
        if not self.do_gng_task:
            self.rules = self.rules[0:2]

        self.current_rule = self.rules[self.rule_block_counter]

        # SAVED PARAMETERS AT THE END OF THE TRIAL
        # reward
        self.reward_mat = []

        # performance
        self.perf_mat = []

        # duration of trial
        self.dur_tr = []

        # stimuli presented
        self.stms_conf = []

        # current task rule
        self.rule_mat = []

        # summed activity across the trial
        self.net_smmd_act = []

        # point by point parameter mats saved for some trials
        self.all_pts_data = data.data(folder=folder)

        # save all points step. Here I call the class data that
        # implements all the necessary functions
        self.sv_pts_stp = 50
        self.num_tr_svd = 100
        self.sv_tr_stp = 5000

        # plot trial events
        try:
            self.plot_trial()
        except IOError:
            print('could not save the figure')

    def get_state(self):
        """
        get_state returns the corresponding state
        needs to check the go/no-go events to know whether to present a
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
            aux = np.random.uniform(low=0., high=self.bg_noise, size=(6,))
        else:
            aux = np.random.normal(stim, self.perc_noise,
                                   size=(int(100*self.Dt),))
            aux = np.histogram(aux, bins=np.arange(7)-0.5)[0]
            aux = aux/np.sum(aux)

        self.state = aux
        # reshape
        self.state = np.reshape(self.state, [1, np.size(self.stims), 1])

        return self.state

    def new_trial(self):
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
        self.internal_state = np.random.choice([0, 1], (3, 1))

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

    def pullArm(self, action, net_state=[]):
        """
        receives an action and returns a reward, a state and flag
        variables that indicate whether to start a new trial
        and whether to update the network
        """
        trial_dur = 0
        # this is whether the RNN does well the dpa task.
        correct_dpa = False
        correct_gng = False
        update_net = False
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
            if action != 0:
                reward = self.rewards[0]
            else:
                reward = self.rewards[1]

        new_trial = self.td == self.t_stp
        if new_trial:
            # current trial info
            self.dur_tr.append(trial_dur)
            self.perf_mat.append([self.correct_gng, self.correct_dpa])
            self.reward_mat.append(reward)
            new_state = None

            # check if it is time to update the network
            update_net = ((self.num_tr-1) % self.update_net_step == 0) and\
                (self.num_tr != 1)

            # point by point parameter mats saved for some periods
            aux = np.floor(self.num_tr / self.num_tr_svd)
            if aux % self.sv_pts_stp == 0:
                self.all_pts_data.update(net_state=net_state, reward=reward,
                                         update_net=update_net,
                                         action=action,
                                         correct=[correct_gng, correct_dpa])

            # during some episodes I save all data points
            aux = np.floor((self.num_tr-1)/self.num_tr_svd)
            aux2 = np.floor(self.num_tr/self.num_tr_svd)
            if aux % self.sv_pts_stp == 0 and\
               aux2 % self.sv_pts_stp == 1:
                self.all_pts_data.save(self.num_tr)
                self.all_pts_data.reset()

        else:
            new_state = self.get_state()
            # during some episodes I save all data points
            aux = np.floor(self.num_tr/self.num_tr_svd)
            if aux % self.sv_pts_stp == 0:
                conf_aux = self.stms_conf[self.num_tr-1]
                corr_aux = [correct_gng, correct_dpa]
                self.all_pts_data.update(new_state=new_state,
                                         net_state=net_state,
                                         reward=reward,
                                         update_net=update_net,
                                         action=action,
                                         correct=corr_aux,
                                         new_trial=new_trial,
                                         num_trials=self.num_tr,
                                         stim_conf=conf_aux)

        return new_state, reward, update_net, new_trial

    def save_trials_data(self):
        # Periodically save model trials statistics.
        if self.num_tr % self.sv_tr_stp == 0:
            data = {'trial_duration': self.dur_tr,
                    'stims_conf': self.stms_conf, 'rule': self.rule_mat,
                    'net_smmd_act': self.net_smmd_act,
                    'reward': self.reward_mat, 'performance': self.perf_mat}
            np.savez(self.folder + '/trials_stats_' + str(self.num_tr) +
                     '.npz', **data)

    def plot_trial(self):
        task = np.zeros((3, self.td))
        x_ticks = []
        x_ticks_labels = []
        prev_st = [0, 0, 0]
        for ind_t in range(1, self.td+1):
            dpa_st1, gng_st, dpa_st2 =\
                stim_periods(ind_t, self.gng_t, self.gng_st, self.dpa_st,
                             self.dpa_d, self.dpa_resp, self.bt_tr_time,
                             self.td)
            gng_resp, dpa_resp, gng_end, dpa_end =\
                response_periods(ind_t, self.gng_t,
                                 self.gng_st, self.gng_d, self.gng_resp,
                                 self.dpa_st, self.dpa_d, self.dpa_resp,
                                 self.bt_tr_time, self.td)
            # current state
            curr_st = [gng_st, 2*(dpa_st1 + dpa_st2), 3*(gng_resp + dpa_resp)]
            task[0, ind_t-1] = curr_st[0]
            task[1, ind_t-1] = curr_st[1]
            task[2, ind_t-1] = curr_st[2]
            # store relevant time stamps
            if (curr_st != prev_st):
                x_ticks.append(ind_t-1.5)
                x_ticks_labels.append(str(ind_t-1))
            prev_st = curr_st

        f = plt.figure(figsize=(8, 10), dpi=250)
        matplotlib.rcParams.update({'font.size': 8})
        plt.subplots_adjust(left=left, bottom=bottom, right=right,
                            top=top, wspace=wspace, hspace=hspace)
        plt.imshow(task)
        plt.yticks([0, 1, 2], ['GNG', 'DPA', 'resp. w.'])
        plt.xticks(x_ticks, x_ticks_labels)
        f.savefig(self.folder[:self.folder.find('trains')] + '/protocol.svg',
                  dpi=600, bbox_inches='tight')

# STIMULUS AND REWARD PERIODS


def rew_deliv_control(gt, action):
    correct = gt[0] == action  # correct when action == ground truth
    flag = action == 0  # the flag is only false when the net goes
    return correct, flag


def stim_periods(t_stp,
                 gng_t, gng_st, dpa_st, dpa_d, dpa_resp,
                 bt_tr_time, td):
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
    start_gng_resp, end_gng_resp = gng_r(gng_t, gng_st, gng_d, gng_resp)
    gng = start_gng_resp < t_stp <= end_gng_resp
    #
    start_dpa_resp, end_dpa_resp = dpa2_r(dpa_resp, bt_tr_time, td)
    dpa = start_dpa_resp < t_stp <= end_dpa_resp

    end_gng_flg = t_stp == end_gng_resp + 1
    end_dpa_flg = t_stp == end_dpa_resp + 1
    return gng, dpa, end_gng_flg, end_dpa_flg


def dpa2_stim(dpa_st, dpa_d, dpa_resp, bt_tr_time, td):
    last_periods = dpa_st+dpa_d+dpa_resp+bt_tr_time
    dpa_2_st = td-last_periods
    dpa_2_end = td-last_periods + dpa_st
    return dpa_2_st, dpa_2_end


def dpa2_r(dpa_resp, bt_tr_time, td):
    last_periods = dpa_resp+bt_tr_time
    start_dpa_resp = td-last_periods
    end_dpa_resp = td-last_periods + dpa_resp
    return start_dpa_resp, end_dpa_resp


def gng_r(gng_t, gng_st, gng_d, gng_resp):
    start_gng_resp = gng_t+gng_st+gng_d
    end_gng_resp = gng_t+gng_st+gng_d+gng_resp
    return start_gng_resp, end_gng_resp


if __name__ == '__main__':
    plt.close('all')
    inst = 0
    exp_dur = int(2.5*10**4)
    gamma = .9
    update_net_step = 2
    Dt = 0.2
    td = 4.0
    gng_time = 0.8
    gng_st = 0.4
    gng_d = 0.2
    gng_resp = 0.4
    dpa_st = 0.6
    dpa_d = 0.4
    dpa_resp = 0.4
    rewards = [-0.1, 0.0, 1.0, -1.0]
    block_dur = 0
    num_units = 32
    bg_noise = 0.001
    perc_noise = 1.
    network = 'relu'
    learning_rate = 0.001
    do_gng_task = True
    bt_tr_t = 0.2
    if block_dur == 0:
        bd = exp_dur+1
    else:
        bd = block_dur
    task = dual_task(update_net_step=update_net_step,
                     Dt=Dt, td=td, gng_time=gng_time,
                     bt_tr_time=bt_tr_t, dpa_st=dpa_st,
                     dpa_d=dpa_d, dpa_resp=dpa_resp,
                     gng_st=gng_st, gng_d=gng_d,
                     gng_resp=gng_resp,
                     rewards=rewards, bg_noise=bg_noise,
                     perc_noise=perc_noise,
                     folder='/home/molano/dual_task_project/',
                     block_dur=bd, do_gng_task=do_gng_task)

    new_trial = True
    st_mat = []
    rew_mat = []
    conf_mat = []
    tr_count = -1
    for ind_stp in range(800):
        if new_trial:
            state = task.new_trial()
            st_mat.append(state[:, :, 0])
            new_trial = False
            tr_count += 1
        else:
            state, reward, _, new_trial = task.pullArm(0, net_state=[])
            rew_mat.append(reward)
            conf_mat.append(task.stms_conf[tr_count])
            if state is not None:
                st_mat.append(state[:, :, 0])

    num_steps = 400
    st_mat = np.array(st_mat)  # np.vstack(
    shape_aux = (st_mat.shape[0], st_mat.shape[2])
    st_mat = np.reshape(st_mat, shape_aux)
    plt.figure(figsize=(8, 4), dpi=250)
    plt.subplot(3, 1, 1)
    plt.imshow(st_mat[:num_steps].T, aspect='auto')

    conf_mat = np.array(conf_mat)
    shape_aux = (conf_mat.shape[0], conf_mat.shape[1])
    conf_mat = np.reshape(conf_mat, shape_aux)
    plt.subplot(3, 1, 2)
    plt.imshow(conf_mat[:num_steps].T, aspect='auto')
