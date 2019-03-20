from scipy.optimize import curve_fit
from scipy.special import erf
import utils as ut
import numpy as np
import matplotlib.pyplot as plt


def plot_learning(performance, evidence, stim_position, action):
    """
    plots RNN and ideal observer performances.
    The function assumes that a figure has been created
    before it is called.
    """
    # remove all previous plots
    ut.rm_lines()
    # ideal observer choice
    io_choice = (evidence < 0) + 1
    io_performance = io_choice == stim_position
    # save the mean performances
    RNN_perf = np.mean(performance[:, 2000:].flatten())
    io_perf = np.mean(io_performance[:, 2000:].flatten())

    w_conv = 200  # this is for the smoothing
    # plot smoothed performance
    performance_smoothed = np.convolve(np.mean(performance, axis=0),
                                       np.ones((w_conv,))/w_conv,
                                       mode='valid')
    plt.plot(performance_smoothed, color=(0.39, 0.39, 0.39), lw=0.5,
             label='RNN perf. (' + str(round(RNN_perf, 3)) + ')')
    print('RNN perf: ' + str(round(RNN_perf, 3)))
    # plot ideal observer performance
    io_perf_smoothed = np.convolve(np.mean(io_performance, axis=0),
                                   np.ones((w_conv,))/w_conv,
                                   mode='valid')
    plt.plot(io_perf_smoothed, color=(1, 0.8, 0.5), lw=0.5,
             label='Ideal Obs. perf. (' + str(round(io_perf, 3)) + ')')
    # plot 0.25, 0.5 and 0.75 performance lines
    plot_fractions([0, performance.shape[1]])
    plt.title('performance')
    plt.xlabel('trials')
    plt.legend()


def plot_fractions(lims):
    """
    plot dashed lines for 0.25, 0.5 and 0.75
    """
    plt.plot(lims, [0.25, 0.25], '--k', lw=0.25)
    plt.plot(lims, [0.5, 0.5], '--k', lw=0.25)
    plt.plot(lims, [0.75, 0.75], '--k', lw=0.25)
    plt.xlim(lims[0], lims[1])


def plot_psychometric_curves(evidence, performance, action,
                             blk_dur=200,
                             plt_av=True, figs=True):
    """
    plots psychometric curves
    - evidence for right VS prob. of choosing right
    - evidence for repeating side VS prob. of repeating
    - same as above but conditionated on hits and fails
    The function assumes that a figure has been created
    before it is called.
    """
    # build the mat that indicates the current block
    blocks = build_block_mat(evidence.shape, blk_dur)

    # repeating probs. values
    probs_vals = np.unique(blocks)
    assert len(probs_vals) <= 2
    colors = [[1, 0, 0], [0, 0, 1]]
    if figs:
        rows = 2
        cols = 2
    else:
        rows = 0
        cols = 0

    data = {}
    for ind_sp in range(4):
        if figs:
            plt.subplot(rows, cols, ind_sp+1)
            # remove all previous plots
            ut.rm_lines()
    for ind_blk in range(len(probs_vals)):
        # filter data
        inds = (blocks == probs_vals[ind_blk])
        evidence_block = evidence[inds]
        performance_block = performance[inds]
        action_block = action[inds]
        data = get_psyCho_curves_data(performance_block,
                                      evidence_block, action_block,
                                      probs_vals[ind_blk],
                                      rows, cols, figs, colors[ind_blk],
                                      plt_av, data)
    return data


def get_psyCho_curves_data(performance, evidence, action, prob,
                           rows, cols, figs, color, plt_av, data):
    """
    plot psychometric curves for:
    right evidence VS prob. choosing right
    repeating evidence VS prob. repeating
    repeating evidence VS prob. repeating (conditionated on previous correct)
    repeating evidence VS prob. repeating (conditionated on previous wrong)
    """

    # 1. RIGHT EVIDENCE VS PROB. CHOOSING RIGHT
    # get the action
    right_choice = action == 1

    # associate invalid trials (network fixates) with incorrect choice
    right_choice[action == 0] = evidence[action == 0] > 0
    # np.random.choice([0, 1], size=(np.sum(action.flatten() == 2),))

    # convert the choice to float and flatten it
    right_choice = [float(x) for x in right_choice]
    right_choice = np.asarray(right_choice)
    # fit and plot
    if figs:
        plt.subplot(rows, cols, 1)
        plt.xlabel('right evidence')
        plt.ylabel('prob. right')
    popt, pcov, av_data =\
        fit_and_plot(evidence, right_choice,
                     plt_av, color=color, figs=figs)

    data['popt_rightProb_' + str(prob)] = popt
    data['pcov_rightProb_' + str(prob)] = pcov
    data['av_rightProb_' + str(prob)] = av_data

    # 2. REPEATING EVIDENCE VS PROB. REPEATING
    # I add a random choice to the beginning of the choice matrix
    # and differentiate to see when the network is repeating sides
    repeat = np.concatenate(
        (np.array(np.random.choice([0, 1])).reshape(1,),
         right_choice))
    repeat = np.diff(repeat) == 0
    # right_choice_repeating is just the original right_choice mat
    # but shifted one element to the left.
    right_choice_repeating = np.concatenate(
        (np.array(np.random.choice([0, 1])).reshape(1, ),
         right_choice[:-1]))
    # the rep. evidence is the original evidence with a negative sign
    # if the repeating side is the left one
    rep_ev_block = evidence *\
        (-1)**(right_choice_repeating == 0)
    # fitting
    if figs:
        label_aux = 'p. rep.: ' + str(prob)
        plt.subplot(rows, cols, 2)
        #         plt.xlabel('repetition evidence')
        #         plt.ylabel('prob. repetition')
    else:
        label_aux = ''
    popt, pcov, av_data =\
        fit_and_plot(rep_ev_block, repeat,
                     plt_av, color=color,
                     label=label_aux, figs=figs)

    data['popt_repProb_'+str(prob)] = popt
    data['pcov_repProb_'+str(prob)] = pcov
    data['av_repProb_'+str(prob)] = av_data

    # plot psycho-curves conditionated on previous performance
    # get previous trial performance
    prev_perf = np.concatenate(
        (np.array(np.random.choice([0, 1])).reshape(1,),
         performance[:-1]))
    # 3. REPEATING EVIDENCE VS PROB. REPEATING
    # (conditionated on previous correct)
    # fitting
    mask = prev_perf == 1
    if figs:
        plt.subplot(rows, cols, 3)
        plt.xlabel('repetition evidence')
        plt.ylabel('prob. repetition')
        #         plt.title('Prev. hit')
    popt, pcov, av_data =\
        fit_and_plot(rep_ev_block[mask], repeat[mask],
                     plt_av, color=color,
                     label=label_aux, figs=figs)

    data['popt_repProb_hits_'+str(prob)] = popt
    data['pcov_repProb_hits_'+str(prob)] = pcov
    data['av_repProb_hits_'+str(prob)] = av_data
    # print('bias: ' + str(round(popt[1], 3)))
    # 4. REPEATING EVIDENCE VS PROB. REPEATING
    # (conditionated on previous wrong)
    # fitting
    mask = prev_perf == 0
    if figs:
        plt.subplot(rows, cols, 4)
        plt.xlabel('repetition evidence')
        #         plt.ylabel('prob. repetition')
        #         plt.title('Prev. fail')
    popt, pcov, av_data =\
        fit_and_plot(rep_ev_block[mask], repeat[mask],
                     plt_av, color=color,
                     label=label_aux, figs=figs)

    data['popt_repProb_fails_'+str(prob)] = popt
    data['pcov_repProb_fails_'+str(prob)] = pcov
    data['av_repProb_fails_'+str(prob)] = av_data

    return data


def fit_and_plot(evidence, choice, plt_av=False,
                 color=(0, 0, 0), label='', figs=False):
    """
    uses curve_fit to fit the evidence/choice provided to a probit function
    that takes into account the lapse rates
    it also plots the corresponding fit and, if plt_av=True, plots the
    average choice values for different windows of the evidence
    """
    if evidence.shape[0] > 10 and len(np.unique(choice)) == 2:
        # fit
        popt, pcov = curve_fit(probit_lapse_rates,
                               evidence, choice, maxfev=10000)
    # plot averages
        if plt_av:
            av_data = plot_psychoCurves_averages(evidence, choice,
                                                 color=color, figs=figs)
        else:
            av_data = {}
        # plot obtained probit function
        if figs:
            x = np.linspace(np.min(evidence),
                            np.max(evidence), 50)
            # get the y values for the fitting
            y = probit_lapse_rates(x, popt[0], popt[1], popt[2], popt[3])
            if label == '':
                plt.plot(x, y, color=color, lw=0.5)
            else:
                plt.plot(x, y, color=color,  label=label
                         + ' b: ' + str(round(popt[1], 3)), lw=0.5)
                plt.legend(loc="lower right")
            plot_dashed_lines(-np.max(evidence), np.max(evidence))
    else:
        av_data = {}
        popt = [0, 0, 0, 0]
        pcov = 0
        print('not enough data!')
    return popt, pcov, av_data


def plot_psychoCurves_averages(x_values, y_values,
                               color=(0, 0, 0), figs=False):
    """
    plots average values of y_values for 10 (num_values) different windows
    in x_values
    """
    num_values = 10
    conf = 0.95
    x, step = np.linspace(np.min(x_values), np.max(x_values),
                          num_values, retstep=True)
    curve_mean = []
    curve_std = []
    # compute mean for each window
    for ind_x in range(num_values-1):
        inds = (x_values >= x[ind_x])*(x_values < x[ind_x+1])
        mean = np.mean(y_values[inds])
        curve_mean.append(mean)
        curve_std.append(conf*np.sqrt(mean*(1-mean)/np.sum(inds)))

    if figs:
        # make color weaker
        # np.max(np.concatenate((color, [1, 1, 1]), axis=0), axis=0)
        color_w = np.array(color) + 0.5
        color_w[color_w > 1] = 1
        # plot
        plt.errorbar(x[:-1] + step / 2, curve_mean, curve_std,
                     color=color_w, marker='+', linestyle='')

    # put values in a dictionary
    av_data = {'mean': curve_mean, 'std': curve_std, 'x': x[:-1]+step/2}
    return av_data


def build_block_mat(shape, block_dur):
    # build rep. prob vector
    rp_mat = np.zeros(shape)
    a = np.arange(shape[1])
    b = np.floor(a/block_dur)
    rp_mat[:, b % 2 == 0] = 1
    return rp_mat


def probit_lapse_rates(x, beta, alpha, piL, piR):
    piR = 0
    piL = 0
    probit_lr = piR + (1 - piL - piR) * probit(x, beta, alpha)
    return probit_lr


def probit(x, beta, alpha):
    probit = 1/2*(1+erf((beta*x+alpha)/np.sqrt(2)))
    return probit


def plot_dashed_lines(minimo, maximo):
    plt.plot([0, 0], [0, 1], '--k', lw=0.2)
    plt.plot([minimo, maximo], [0.5, 0.5], '--k', lw=0.2)


def plot_trials(folder, num_steps, num_trials):
    num_rows = 7
    for ind_sp in range(num_rows):
        plt.subplot(num_rows, 1, ind_sp+1)
        # remove all previous plots
        ut.rm_lines()
    data = np.load(folder + '/all_points_' + str(num_trials) + '.npz')
    new_tr_flag = data['new_trial_flags']
    # plot the stimulus
    plt.subplot(num_rows, 1, 1)
    states = data['states']
    shape_aux = (states.shape[0], states.shape[2])
    states = np.reshape(states, shape_aux)[0:num_steps, :]
    plt.imshow(states[:, 0:2].T, aspect='auto', cmap='gray')
    minimo = -0.5
    maximo = 1.5
    ut.plot_trials_start(new_tr_flag, minimo, maximo, num_steps, color='y')

    plt.ylabel('stim')
    plt.xticks([])
    plt.yticks([])

    # go over trials and compute cumulative evidence
    trials = np.nonzero(new_tr_flag)[0]
    trials = np.concatenate((np.array([-1]), trials))
    plt.subplot(num_rows, 1, 2)
    evidence = np.zeros((num_steps,))
    for ind_time in range(num_steps):
        if ind_time in trials:
            # the cumulative evidence is 0 in the beginning
            evidence[ind_time] = 0
        else:
            # 0 belongs to trials so this is always valid
            previous_evidence = evidence[ind_time-1]
            evidence[ind_time] = previous_evidence +\
                (states[ind_time, 0]-states[ind_time, 1])

    # plot evidence
    plt.plot(evidence)
    plt.plot([0, num_steps], [0, 0], '--k', lw=0.5)
    plt.xlim(-0.5, 99.5)
    ut.plot_trials_start(new_tr_flag, np.min(evidence), np.max(evidence),
                         num_steps)
    plt.ylabel('evidence')
    plt.xticks([])
    plt.plot(new_tr_flag)
    # plot actions
    actions = data['actions']
    actions = np.reshape(actions, (1, -1))
    minimo = -0.5
    maximo = 0.5
    plt.subplot(num_rows, 1, 3)
    plt.imshow(actions, aspect='auto', cmap='viridis')
    ut.plot_trials_start(new_tr_flag, minimo, maximo, num_steps, color='w')
    plt.ylabel('action')
    plt.xticks([])
    plt.yticks([])
    # plot the rewards
    rewards = data['rewards']
    rewards = np.reshape(rewards, (1, -1))
    minimo = -0.5
    maximo = 0.5
    plt.subplot(num_rows, 1, 4)
    plt.imshow(rewards, aspect='auto', cmap='jet')
    ut.plot_trials_start(new_tr_flag, minimo, maximo, num_steps, color='w')
    plt.ylabel('reward')
    plt.xticks([])
    plt.yticks([])

    # plot the performance
    performance = np.array(data['corrects'])[0:num_steps, :]
    performance = performance.T
    minimo = -0.5
    maximo = 0.5
    plt.subplot(num_rows, 1, 5)
    plt.imshow(performance, aspect='auto', cmap='jet')
    ut.plot_trials_start(new_tr_flag, minimo, maximo, num_steps, color='w')
    plt.ylabel('correct')
    plt.xticks([])
    plt.yticks([])

    # plot the ground truth
    plt.subplot(num_rows, 1, 6)
    states = data['stims_conf'] == 1.0
    states = states[0:num_steps, :]
    plt.imshow(states[:, 0:2].T, aspect='auto', cmap='gray')
    minimo = -0.5
    maximo = 1.5
    ut.plot_trials_start(new_tr_flag, minimo, maximo, num_steps, color='y')

    plt.ylabel('gr. truth')
    plt.xticks([])
    plt.yticks([])

    # plot neurons' activities
    if len(data['net_state']) != 0:
        activity = data['net_state']
        plt.subplot(num_rows, 1, 7)
        shape_aux = (activity.shape[0], activity.shape[2])
        activity = np.reshape(activity, shape_aux)[0:num_steps, :]
        maximo = np.max(activity, axis=0).reshape(1, activity.shape[1])
        activity /= maximo
        activity[np.isnan(activity)] = -0.1
        plt.imshow(activity.T, aspect='auto', cmap='hot')
        minimo = np.min(-0.5)
        maximo = np.max(shape_aux[1]-0.5)
        ut.plot_trials_start(new_tr_flag, minimo, maximo, num_steps)
        plt.ylabel('activity')
        plt.xlabel('time (a.u)')
        plt.yticks([])


def neural_analysis():
    data = np.load('/home/linux/network_data_132999.npz')
    upd = 10
    env = 0
    plt.figure()
    plt.subplot(4, 1, 1)
    states = data['states'][upd, :, env, :].T
    maxs = np.max(states, axis=1).reshape((128, 1))
    states_norm = states / maxs
    states_norm[np.where(maxs == 0), :] = 0
    plt.imshow(states_norm, aspect='auto')
    plt.subplot(4, 1, 3)
    aux = np.reshape(data['rewards'], (1000, 12, 100))
    plt.plot(aux[upd, env, :])
    plt.xlim([-0.5, 99.5])
    plt.subplot(4, 1, 4)
    aux = np.reshape(data['actions'], (1000, 12, 100))
    plt.plot(aux[upd, env, :])
    plt.xlim([-0.5, 99.5])
    plt.subplot(4, 1, 2)
    print(data['obs'].shape)
    aux = np.reshape(data['obs'], (1000, 12, 100, 3))
    plt.imshow(aux[upd, env, :, :].T, aspect='auto')


if __name__ == '__main__':
    plt.close('all')
    #    neural_analysis()
    #    asdasd
    # ['choice', 'stimulus', 'correct_side', 'obs_mat', 'act_mat', 'rew_mat']
    # obs_mat = data['obs_mat']
    # act_mat = data['act_mat']
    # rew_mat = data['rew_mat']
    data = np.load('/home/linux/TrialHistory0_data.npz')
    # data = np.load('/home/linux/RDM0_data.npz')
    choice = data['choice']
    stimulus = data['stimulus']
    correct_side = data['correct_side']
    correct_side = np.reshape(correct_side, (1, len(correct_side)))
    choice = np.reshape(choice, (1, len(choice)))
    print(choice.shape[1])
    correct_side[np.where(correct_side == -1)] = 2
    correct_side = np.abs(correct_side-3)
    performance = (choice == correct_side)
    evidence = stimulus[:, 1] - stimulus[:, 2]
    evidence = np.reshape(evidence, (1, len(evidence)))
    # plot performance
    num_tr = 300000
    start_point = 50000
    plt.figure()
    plot_learning(performance[:, start_point:start_point+num_tr],
                  evidence[:, start_point:start_point+num_tr],
                  correct_side[:, start_point:start_point+num_tr],
                  choice[:, start_point:start_point+num_tr])

    # plot performance last training stage
    plt.figure()
    num_tr = 20000
    start_point = performance.shape[1]-num_tr
    plot_learning(performance[:, start_point:start_point+num_tr],
                  evidence[:, start_point:start_point+num_tr],
                  correct_side[:, start_point:start_point+num_tr],
                  choice[:, start_point:start_point+num_tr])
    # plot trials
    correct_side_plt = correct_side[:, :400]
    plt.figure()
    plt.imshow(correct_side_plt, aspect='auto')

    # compute bias across training
    per = 1000000
    num_stps = int(choice.shape[1] / per)
    bias_0_hit = []
    bias_1_hit = []
    bias_0_fail = []
    bias_1_fail = []
    for ind_per in range(num_stps):
        ev = evidence[:, ind_per*per:(ind_per+1)*per]
        perf = performance[:, ind_per*per:(ind_per+1)*per]
        ch = choice[:, ind_per*per:(ind_per+1)*per]
        data = plot_psychometric_curves(ev, perf, ch, blk_dur=200,
                                        plt_av=False, figs=False)
        bias_0_hit.append(data['popt_repProb_hits_0.0'][1])
        bias_1_hit.append(data['popt_repProb_hits_1.0'][1])
        bias_0_fail.append(data['popt_repProb_fails_0.0'][1])
        bias_1_fail.append(data['popt_repProb_fails_1.0'][1])

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(bias_0_hit)
    plt.title('block 0 after correct')

    plt.subplot(2, 2, 2)
    plt.plot(bias_1_hit)
    plt.title('block 1 after correct')

    plt.subplot(2, 2, 3)
    plt.plot(bias_0_fail)
    plt.title('block 0 after error')

    plt.subplot(2, 2, 4)
    plt.plot(bias_1_fail)
    plt.title('block 1 after error')
    # plot psychometric curves
    plt.figure()
    plot_psychometric_curves(evidence, performance, choice,
                             blk_dur=200, plt_av=True, figs=True)
