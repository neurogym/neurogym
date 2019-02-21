import numpy as np
import data
import matplotlib.pyplot as plt
import analysis as an


class PriorsEnv():
    """
    two-alternative forced choice task where the probability of repeating the 
    previous choice is parametrized
    """
    def __init__(self, trial_dur=10, upd_net=5, rep_prob=(.2, .8),
                 rewards=(0.1, -0.1, 1.0, -1.0),
                 env_seed='0', block_dur=200, stim_ev=0.5, folder=None,
                 plot=False):
        # num steps per trial
        self.trial_dur = trial_dur
        # rewards given for: stop fixating, keep fixating, correct, wrong
        self.rewards = rewards
        # number of trials per blocks
        self.block_dur = block_dur
        # stimulus evidence: one stimulus is always N(1,1), the mean of
        # the other is drawn from a uniform distrib.=U(stim_ev,1).
        # stim_ev must then be between 0 and 1 and the higher it is
        # the more difficult will be the task
        self.stim_ev = stim_ev
        # prob. of repeating the stimuli in the positions of previous trial
        self.rep_prob = rep_prob
        # model instance
        self.env_seed = env_seed
        # folder to save data
        self.folder = folder
        # update parameters
        self.upd_net = upd_net

        # num actions
        self.num_actions = 3

        # position of the first stimulus
        self.stms_pos_new_trial = np.random.choice([0, 1])
        # keeps track of the repeating prob of the current block
        self.curr_rep_prob = np.random.choice([0, 1])
        # position of the stimuli
        self.stm_pos_new_trial = 0
        # steps counter
        self.timestep = 0
        # initialize ground truth state [stim1 mean, stim2 mean, fixation])
        # the network has to output the action corresponding to the stim1 mean
        # that will be always 1.0 (I just initialize here at 0 for convinience)
        self.int_st = np.array([0, 0, -1])
        # accumulated evidence
        self.evidence = 0
        # number of trials
        self.num_tr = 0

        # trial data to save
        # stimulus evidence
        self.ev_mat = []
        # position of stimulus 1
        self.stm_pos = []
        # performance
        self.perf_mat = []
        # summed activity across the trial
        self.action = []
        # current repeating probability
        self.rp_mat = []
        

        # point by point parameter mats saved for some trials
        self.all_pts_data = data.data(folder=folder)

        # save all points step. Here I call the class data that implements
        # all the necessary functions
        self.sv_pts_stp = 10
        self.num_tr_svd = 1000

        # figures for plotting
        self.plot_figs = plot
        if self.plot_figs:
            self.perf_fig = plt.figure(figsize=(8, 8), dpi=100)
            self.bias_fig = plt.figure(figsize=(8, 8), dpi=100)
            self.trials_fig = plt.figure(figsize=(8, 8), dpi=100)

            print('--------------- Priors experiment ---------------')
            print('Update of networks (in trials): ' +
                  str(self.upd_net))
            print('Duration of each trial (in steps): ' + str(self.trial_dur))
            print('Rewards: ' + str(self.rewards))
            print('Duration of each block (in trials): ' + str(self.block_dur))
            print('Repeating probabilities of each block: ' +
                  str(self.rep_prob))
            print('Stim evidence: ' + str(self.stim_ev))
            print('Saving folder: ' + str(self.folder))
            print('--------------- ----------------- ---------------')

    def step(self, action, net_st=[]):
        """
        receives an action and returns a reward, a state and flag variables
        that indicate whether to start a new trial and whether to update
        the network
        """
        new_trial = True
        correct = False
        done = False
        # decide which reward and state (new_trial, correct) we are in
        if self.timestep < self.trial_dur:
            if (self.int_st[action] != -1).all():
                reward = self.rewards[0]
            else:
                # don't abort the trial even if the network stops fixating
                reward = self.rewards[1]

            new_trial = False

        else:
            if (self.int_st[action] == 1.0).all():
                reward = self.rewards[2]
                correct = True
            else:
                reward = self.rewards[3]

        if new_trial:
            # keep main variables of the trial
            self.stm_pos.append(self.stms_pos_new_trial)
            self.perf_mat.append(correct)
            self.action.append(action)
            self.ev_mat.append(self.evidence)
            self.rp_mat.append(self.rep_prob)
            new_st = self.new_trial()
            # check if it is time to update the network
            done = ((self.num_tr-1) % self.upd_net == 0) and (self.num_tr != 1)
            # check if it is time to save the trial-to-trial data
            if self.num_tr % 10000 == 0:
                self.save_trials_data()
                if self.plot_figs:
                    self.output_stats()

            # during some episodes I save all data points
            aux = np.floor((self.num_tr-1) / self.num_tr_svd)
            aux2 = np.floor(self.num_tr / self.num_tr_svd)
            if aux % self.sv_pts_stp == 0 and\
               aux2 % self.sv_pts_stp == 1:
                self.all_pts_data.save(self.num_tr)
                self.all_pts_data.reset()
                if self.plot_figs:
                    plt.figure(self.trials_fig.number)
                    an.plot_trials(self.folder, 100, self.num_tr)
                    self.trials_fig.canvas.draw()

        else:
            new_st = self.get_state()
        
        
        # during some episodes I save all data points
        if np.floor(self.num_tr / self.num_tr_svd) % self.sv_pts_stp == 0:
            self.all_pts_data.update(new_state=new_st, net_state=net_st,
                                     reward=reward, update_net=done,
                                     action=action, correct=[correct],
                                     new_trial=new_trial,
                                     num_trials=self.num_tr,
                                     stim_conf=self.int_st)

        return new_st, reward, done, new_trial

    def get_state(self):
        """
        Outputs a new observation using stim 1 and 2 means.
        It also outputs a fixation signal that is always -1 except at the
        end of the trial that is 0
        """
        self.timestep += 1
        # if still in the integration period present a new observation
        if self.timestep < self.trial_dur:
            self.state = [np.random.normal(self.int_st[0]),
                          np.random.normal(self.int_st[1]), -1]
        else:
            self.state = [0, 0, 0]

        # update evidence
        self.evidence += self.state[0]-self.state[1]

        return np.reshape(self.state, [1, self.num_actions, 1])

    def new_trial(self):
        """
        this function creates a new trial, deciding the amount of coherence
        (through the mean of stim 2) and the position of stim 1. Once it has
        done this it calls get_state to get the first observation of the trial
        """
        self.num_tr += 1
        self.timestep = 0
        self.evidence = 0
        # this are the means of the two stimuli
        stim1 = 1.0
        stim2 = np.random.uniform(1-self.stim_ev, 1)
        assert stim2 != 1.0
        self.choices = [stim1, stim2]

        # decide the position of the stims
        # if the block is finished update the prob of repeating
        if self.num_tr % self.block_dur == 0:
            self.curr_rep_prob = int(not self.curr_rep_prob)

        # flip a coin
        repeat = np.random.uniform() < self.rep_prob[self.curr_rep_prob]
        if not repeat:
            self.stms_pos_new_trial = not(self.stms_pos_new_trial)

        aux = [self.choices[x] for x in [int(self.stms_pos_new_trial),
                                         int(not self.stms_pos_new_trial)]]

        self.int_st = np.concatenate((aux, np.array([-1])))

        # get state
        s = self.get_state()

        return s

    def save_trials_data(self):
        """
        save trial-to-trial data for:
        evidence, stim postion, action taken and outcome
        """
        # Periodically save model trials statistics.
        data = {'stims_position': self.stm_pos,
                'action': self.action,
                'performance': self.perf_mat,
                'evidence': self.ev_mat,
                'rep_prob': self.rp_mat}
        np.savez(self.folder + '/trials_stats_' +
                 str(self.env_seed) + '_' + str(self.num_tr) + '.npz', **data)

    def reset(self):
        return self.new_trial()

    def output_stats(self):
        """
        plot temporary learning and bias curves
        """
        aux_shape = (1, len(self.ev_mat))
        # plot psycho. curves
        plt.figure(self.bias_fig.number)
        per = 20000
        ev = self.ev_mat.copy()
        ev = np.reshape(ev, aux_shape)[np.max([0, len(ev)-per]):]
        perf = self.perf_mat.copy()
        perf = np.reshape(perf, aux_shape)[np.max([0, len(perf)-per]):]
        action = self.action.copy()
        action = np.reshape(action, aux_shape)[np.max([0, len(action)-per]):]
        an.plot_psychometric_curves(ev, perf, action, blk_dur=self.block_dur)
        self.bias_fig.canvas.draw()
        # plot learning
        plt.figure(self.perf_fig.number)
        ev = self.ev_mat.copy()
        ev = np.reshape(ev, aux_shape)[np.max([0, len(ev)-per]):]
        perf = self.perf_mat.copy()
        perf = np.reshape(perf, aux_shape)[np.max([0, len(perf)-per]):]
        action = self.action.copy()
        action = np.reshape(action, aux_shape)[np.max([0, len(action)-per]):]
        stim_pos = self.stm_pos.copy()
        stim_pos = np.reshape(stim_pos,
                              aux_shape)[np.max([0, len(stim_pos)-per]):]
        an.plot_learning(perf, ev, stim_pos, action)
        self.perf_fig.canvas.draw()
        print('--------------------')