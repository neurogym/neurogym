from scipy.optimize import curve_fit
from scipy.special import erf
import utils as ut
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sstats


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

# NEW YORK PROJECT ##############################################


def bias_calculation(choice, ev, mask):
    # associate invalid trials (network fixates) with incorrect choice
    choice[choice == 0] = ev[choice == 0] > 0
    repeat = np.concatenate(
        (np.array(np.random.choice([0, 1])).reshape(1,),
         choice))
    repeat = np.diff(repeat) == 0
    # right_choice_repeating is just the original right_choice mat
    # but shifted one element to the left.
    choice_repeating = np.concatenate(
        (np.array(np.random.choice([0, 1])).reshape(1, ),
         choice[:-1]))
    # the rep. evidence is the original evidence with a negative sign
    # if the repeating side is the left one
    rep_ev = ev *\
        (-1)**(choice_repeating == 2)
    rep_ev_mask = rep_ev[mask]
    repeat_mask = repeat[mask]
    popt, pcov = curve_fit(probit_lapse_rates, rep_ev_mask, repeat_mask,
                           maxfev=10000)
    return popt, pcov


def neural_analysis(file='/home/linux/network_data_492999.npz', fig=False):
    data = np.load(file)
    env = 0
    rows = 4
    cols = 1
    num_steps = 100
    # states
    states = data['states'][:, :, env, :]

    states = np.reshape(np.transpose(states, (2, 0, 1)),
                        (states.shape[2], np.prod(states.shape[0:2])))
    # actions
    # separate into diff. envs
    actions = np.reshape(data['actions'], (1000, 12, 100))
    # select env.
    actions = actions[:, env, :]
    # flatten
    actions = actions.flatten()
    actions = np.concatenate((np.array([0]), actions[:-1]))
    # obs and rewards (rewards are passed as part of the observation)
    obs = np.reshape(data['obs'], (1000, 12, 100, 4))
    obs = obs[:, env, :, :]
    obs = np.reshape(np.transpose(obs, (2, 0, 1)),
                     (obs.shape[2], np.prod(obs.shape[0:2])))
    rewards = obs[3, :]
    rewards = rewards.flatten()
    ev = obs[1, :] - obs[2, :]
    ev = ev.flatten()
    # trials
    trials = np.reshape(data['trials'], (1000, 12, 100))
    trials = trials[:, env, :]
    trials = np.abs(trials.flatten() - int(1))
    trials = np.concatenate((np.array([0]), trials[:-1]))

    if fig:
        ut.get_fig()
        # FIGURE
        # states
        plt.subplot(rows, cols, 2)
        maxs = np.max(states, axis=1).reshape((128, 1))
        states_norm = states / maxs
        states_norm[np.where(maxs == 0), :] = 0
        plt.imshow(states_norm[:, 0:num_steps], aspect='auto')
        # actions
        plt.subplot(rows, cols, 3)
        plt.plot(actions[0:num_steps], '-+')
        plt.xlim([-0.5, num_steps-0.5])
        # obs
        plt.subplot(rows, cols, 1)
        plt.imshow(obs[:, 0:num_steps], aspect='auto')
        # trials
        plt.subplot(rows, cols, 4)
        plt.plot(trials[0:num_steps], '-+')
        plt.xlim([-0.5, num_steps-0.5])

    return states, rewards, actions, ev, trials


def neuron_selectivity(activity, feature, all_times, feat_bin=None,
                       window=(-5, 10)):
    times = all_times[np.logical_and(all_times > np.abs(window[0]),
                                     all_times < all_times.shape[0]-window[1])]
    feat_mat = feature[times]
    act_mat = []
    for ind_t in range(times.shape[0]):
        start = times[ind_t]+window[0]
        end = times[ind_t]+window[1]
        act_mat.append(activity[start:end])

    if feat_bin is not None:
        feat_mat_bin = np.ceil(feat_bin*(feat_mat-np.min(feat_mat)+1e-5) /
                               (np.max(feat_mat)-np.min(feat_mat)+2e-5))
        feat_mat_bin = feat_mat_bin / feat_bin
    else:
        feat_mat_bin = feat_mat

    values = np.unique(feat_mat_bin)
    act_mat = np.array(act_mat)
    resp_mean = []
    resp_std = []
    significance = []
    for ind_f in range(values.shape[0]):
        feat_resps = act_mat[feat_mat_bin == values[ind_f], :]
        resp_mean.append(np.mean(feat_resps, axis=0))
        resp_std.append(np.std(feat_resps, axis=0) /
                        np.sqrt(feat_resps.shape[0]))
        for ind_f2 in range(ind_f+1, values.shape[0]):
            feat_resps2 = act_mat[feat_mat_bin == values[ind_f2], :]
            for ind_t in range(feat_resps.shape[1]):
                _, pvalue = sstats.ranksums(feat_resps[:, ind_t],
                                            feat_resps2[:, ind_t])
                significance.append([ind_f, ind_f2, ind_t, pvalue])

    return resp_mean, resp_std, values, significance


def get_psths(states, feature, times, window, index, feat_bin=None,
              pv_th=0.01, sorting=None):
    means_neurons = []
    stds_neurons = []
    significances = []
    for ind_n in range(states.shape[0]):
        sts_n = states[ind_n, :]
        means, stds, values, sign =\
            neuron_selectivity(sts_n, feature, times, feat_bin=feat_bin,
                               window=window)
        sign = np.array(sign)
        perc_sign = 100*np.sum(sign[:, 3] <
                               pv_th / sign.shape[0]) / sign.shape[0]
        significances.append(perc_sign)
        means_neurons.append(means)
        stds_neurons.append(stds)
        if ind_n == 0:
            print('values:')
            print(values)
    if sorting is None:
        sorting = np.argsort(-np.array(significances))
    means_neurons = np.array(means_neurons)
    means_neurons = means_neurons[sorting, :, :]
    stds_neurons = np.array(stds_neurons)
    stds_neurons = stds_neurons[sorting, :, :]

    return means_neurons, stds_neurons, values, sorting


def plot_psths(means_mat, stds_mat, values, neurons, suptit=''):
    f = ut.get_fig()
    for ind_n in range(np.min([30, means_mat.shape[0]])):
        means = means_mat[ind_n, :, :]
        stds = stds_mat[ind_n, :, :]
        plt.subplot(6, 5, ind_n+1)
        for ind_plt in range(values.shape[0]):
            if ind_n == 0:
                plt.errorbar(index, means[ind_plt, :], stds[ind_plt, :],
                             label=str(values[ind_plt]))
            else:
                plt.errorbar(index, means[ind_plt, :], stds[ind_plt, :])
            plt.title(str(neurons[ind_n]))
        if ind_n == 0:
            f.legend()
    f.suptitle(suptit)


def plot_cond_psths(means_mat1, stds_mat1, means_mat2, stds_mat2, values,
                    neurons, suptit=''):
    f = ut.get_fig()
    for ind_n in range(np.min([15, means_mat1.shape[0]])):
        means = means_mat1[ind_n, :, :]
        stds = stds_mat1[ind_n, :, :]
        plt.subplot(6, 5, 2*5*np.floor(ind_n/5) + ind_n % 5 + 1)
        for ind_plt in range(values.shape[0]):
            if ind_n == 0:
                plt.errorbar(index, means[ind_plt, :], stds[ind_plt, :],
                             label=str(values[ind_plt]))
            else:
                plt.errorbar(index, means[ind_plt, :], stds[ind_plt, :])
            plt.title(str(neurons[ind_n]))
            ax = plt.gca()
            ut.color_axis(ax, color='g')
        if ind_n == 0:
            f.legend()

    for ind_n in range(np.min([15, means_mat1.shape[0]])):
        means = means_mat2[ind_n, :, :]
        stds = stds_mat2[ind_n, :, :]
        plt.subplot(6, 5, 5*(2*np.floor(ind_n/5)+1) + ind_n % 5 + 1)
        for ind_plt in range(values.shape[0]):
            plt.errorbar(index, means[ind_plt, :], stds[ind_plt, :])
            ax = plt.gca()
            ut.color_axis(ax, color='r')

    f.suptitle(suptit)


def get_transition_mat(choice, times=None, num_steps=None, conv_window=5):
    # selectivity to transition probability
    rand_choice = np.array(np.random.choice([1, 2])).reshape(1,)
    choice = np.concatenate((rand_choice, choice))
    repeat = (np.diff(choice) == 0)*1.0
    transition = np.convolve(repeat, np.ones((conv_window,)),
                             mode='full')[0:-conv_window+1]
    if times is not None:
        trans_mat = np.zeros((num_steps,))
        print(trans_mat.shape)
        print(times.shape)
        for ind_t in range(times.shape[0]):
            trans_mat[times[ind_t]] = transition[ind_t]
        return trans_mat
    else:
        return transition


if __name__ == '__main__':
    plt.close('all')
    neural_analysis_flag = True
    transition_analysis_flag = True
    bias_analysis_flag = True
    behavior_analysis_flag = True
    test_bias_flag = False
    bias_cond_on_history_flag = True
    if neural_analysis_flag:
        states, rewards, actions, obs, trials = neural_analysis()

        dt = 100
        window = (-5, 10)
        win_l = int(np.diff(window))
        index = np.linspace(dt*window[0], dt*window[1],
                            int(win_l), endpoint=False).reshape((win_l, 1))

        times = np.where(trials == 1)[0]
        # actions
        print('selectivity to actions')
        means_neurons, stds_neurons, values, sorting = get_psths(states,
                                                                 actions,
                                                                 times, window,
                                                                 index)
        plot_psths(means_neurons, stds_neurons, values, sorting,
                   suptit='selectivity to actions')

        # rewards
        print('selectivity to reward')
        means_neurons, stds_neurons, values, sorting = get_psths(states,
                                                                 rewards,
                                                                 times, window,
                                                                 index)
        plot_psths(means_neurons, stds_neurons, values, sorting,
                   suptit='selectivity to reward')

        # obs
        print('selectivity to cumulative observation')
        obs_cum = np.zeros_like(obs)
        for ind_t in range(times.shape[0]):
            if ind_t == 0:
                obs_cum[times[ind_t]] = np.sum(obs[0: times[ind_t]])
            else:
                obs_cum[times[ind_t]] = np.sum(obs[times[ind_t-1]:
                                                   times[ind_t]])
        means_neurons, stds_neurons, values, sorting = get_psths(states,
                                                                 obs_cum,
                                                                 times,
                                                                 window,
                                                                 index,
                                                                 feat_bin=4)
        plot_psths(means_neurons, stds_neurons, values, sorting,
                   suptit='selectivity to cumulative observation')

        print('selectivity to action conditioned on reward')
        window = (-5, 15)
        win_l = int(np.diff(window))
        index = np.linspace(dt*window[0], dt*window[1],
                            int(win_l), endpoint=False).reshape((win_l, 1))
        times_r = times[np.where(rewards[times] == 1)]
        means_r, stds_r, values_r, sorting = get_psths(states, actions,
                                                       times_r, window, index)

        times_nr = times[np.where(rewards[times] == 0)]
        means_nr, stds_nr, values_nr, _ = get_psths(states, actions, times_nr,
                                                    window, index,
                                                    sorting=sorting)
        assert (values_r == values_nr).all()
        plot_cond_psths(means_r, stds_r, means_nr, stds_nr,
                        values_r, sorting,
                        suptit='selectivity to action conditioned on reward')

    if transition_analysis_flag:
        dt = 100
        window = (-5, 10)
        win_l = int(np.diff(window))
        index = np.linspace(dt*window[0], dt*window[1],
                            int(win_l), endpoint=False).reshape((win_l, 1))
        states, rewards, actions, obs, trials = neural_analysis()
        times = np.where(trials == 1)[0]
        choice = actions[times]
        num_steps = trials.shape[0]
        trans_mat = get_transition_mat(choice, times, num_steps=num_steps,
                                       conv_window=4)
        print('selectivity to number of repetitions')
        means_neurons, stds_neurons, values, sorting = get_psths(states,
                                                                 trans_mat,
                                                                 times, window,
                                                                 index)
        plot_psths(means_neurons, stds_neurons, values, sorting,
                   suptit='selectivity to number of repetitions')

        print('selectivity to num of repetitions conditioned on prev. reward')
        rews = np.where(rewards[times] == 1)[0]+1
        times_prev_r = times[rews[:-1]]
        means_r, stds_r, values_r, sorting = get_psths(states, trans_mat,
                                                       times_prev_r, window,
                                                       index)
        non_rews = np.where(rewards[times] == 1)[0]+1
        times_prev_nr = times[non_rews[:-1]]
        means_nr, stds_nr, values_nr, _ = get_psths(states, trans_mat,
                                                    times_prev_nr, window,
                                                    index, sorting=sorting)
        assert (values_r == values_nr).all()
        plot_cond_psths(means_r, stds_r, means_nr, stds_nr,
                        values_r, sorting,
                        suptit='selectivity to num reps cond. on prev. reward')
    if bias_analysis_flag:
        dt = 100
        window = (-5, 10)
        win_l = int(np.diff(window))
        index = np.linspace(dt*window[0], dt*window[1],
                            int(win_l), endpoint=False).reshape((win_l, 1))
        states, rewards, actions, obs, trials = neural_analysis()
        times = np.where(trials == 1)[0]
        choice = actions[times]
        num_steps = trials.shape[0]
        trans_mat = get_transition_mat(choice, times, num_steps=num_steps,
                                       conv_window=2)
        rand_choice = np.array(np.random.choice([1, 2])).reshape(1,)
        previous_choice = np.concatenate((rand_choice, choice[:-1]))
        previous_choice[np.where(previous_choice == 2)] = -1
        bias_mat = trans_mat.copy()
        prev_choice_mat = np.zeros_like(trans_mat)
        choice_mat = np.zeros_like(trans_mat)
        for ind_t in range(times.shape[0]):
            bias_mat[times[ind_t]] *= previous_choice[ind_t]
            prev_choice_mat[times[ind_t]] = previous_choice[ind_t]
            choice_mat[times[ind_t]] = choice[ind_t]
        choice_mat[np.where(choice_mat == 2)] = -1
        ut.get_fig()
        plt.plot(trans_mat[1:100], label='trans-mat')
        plt.plot(bias_mat[1:100], label='bias-mat')
        plt.plot(prev_choice_mat[1:100], '--', label='prev-choice-mat')
        plt.plot(choice_mat[1:100]*0.5, '--', label='choice-mat')
        plt.legend()
        print('selectivity to bias')
        means_neurons, stds_neurons, values, sorting = get_psths(states,
                                                                 bias_mat,
                                                                 times, window,
                                                                 index)
        plot_psths(means_neurons, stds_neurons, values, sorting,
                   suptit='selectivity to bias')
        print('selectivity to bias conditioned on reward')
        rews = np.where(rewards[times] == 1)[0]+1
        times_prev_r = times[rews[:-1]]
        means_r, stds_r, values_r, sorting = get_psths(states, bias_mat,
                                                       times_prev_r, window,
                                                       index)
        non_rews = np.where(rewards[times] == 0)[0]+1
        times_prev_nr = times[non_rews[:-1]]
        means_nr, stds_nr, values_nr, _ = get_psths(states, bias_mat,
                                                    times_prev_nr, window,
                                                    index, sorting=sorting)
        assert (values_r == values_nr).all()
        plot_cond_psths(means_r, stds_r, means_nr, stds_nr,
                        values_r, sorting,
                        suptit='selectivity to bias cond. on reward')

    if behavior_analysis_flag:
        # ['choice', 'stimulus', 'correct_side',
        #  'obs_mat', 'act_mat', 'rew_mat']
        # obs_mat = data['obs_mat']
        # act_mat = data['act_mat']
        # rew_mat = data['rew_mat']
        data = np.load('/home/linux/PassReward0_data.npz')
        # data = np.load('/home/linux/PassReward0_data.npz')
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
        f = ut.get_fig()
        plot_learning(performance[:, start_point:start_point+num_tr],
                      evidence[:, start_point:start_point+num_tr],
                      correct_side[:, start_point:start_point+num_tr],
                      choice[:, start_point:start_point+num_tr])

        # plot performance last training stage
        f = ut.get_fig()
        num_tr = 20000
        start_point = performance.shape[1]-num_tr
        plot_learning(performance[:, start_point:start_point+num_tr],
                      evidence[:, start_point:start_point+num_tr],
                      correct_side[:, start_point:start_point+num_tr],
                      choice[:, start_point:start_point+num_tr])
        # plot trials
        correct_side_plt = correct_side[:, :400]
        f = ut.get_fig()
        plt.imshow(correct_side_plt, aspect='auto')

        # compute bias across training
        per = 100000
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

        f = ut.get_fig()
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

    if test_bias_flag:
        data = np.load('/home/linux/PassReward0_data.npz')
        choice = data['choice']
        stimulus = data['stimulus']
        correct_side = data['correct_side']
        correct_side = np.reshape(correct_side, (1, len(correct_side)))
        choice = np.reshape(choice, (1, len(choice)))
        correct_side[np.where(correct_side == -1)] = 2
        correct_side = np.abs(correct_side-3)
        performance = (choice == correct_side)
        evidence = stimulus[:, 1] - stimulus[:, 2]
        evidence = np.reshape(evidence, (1, len(evidence)))
        # plot psychometric curves
        f = ut.get_fig()
        num_tr = 2000000
        start_point = performance.shape[1]-num_tr
        ev = evidence[:, start_point:start_point+num_tr]
        perf = performance[:, start_point:start_point+num_tr]
        ch = choice[:, start_point:start_point+num_tr]
        plot_psychometric_curves(ev, perf, ch, blk_dur=200,
                                 plt_av=True, figs=True)
        # REPLICATE RESULTS
        # build the mat that indicates the current block
        blocks = build_block_mat(ev.shape, block_dur=200)
        rand_perf = np.array(np.random.choice([0, 1])).reshape(1,)
        prev_perf = np.concatenate((rand_perf, perf[0, :-1]))
        ch = np.reshape(ch, (ch.shape[1],))
        ev = np.reshape(ev, (ev.shape[1],))
        labels = ['error', 'correct']
        for ind_perf in reversed(range(2)):
            for ind_bl in range(2):
                mask = np.logical_and(blocks == ind_bl, prev_perf == ind_perf)
                mask = mask.reshape((mask.shape[1],))
                assert np.sum(mask) > 0
                popt, pcov = bias_calculation(ch, ev, mask)
                print('bias block ' + str(ind_bl) + ' after ' +
                      labels[ind_perf] + ':')
                print(popt[1])

    if bias_cond_on_history_flag:
        data = np.load('/home/linux/PassReward0_data.npz')
        choice = data['choice']
        print('number of trials: ' + str(choice.shape[0]))
        stimulus = data['stimulus']
        correct_side = data['correct_side']
        correct_side = np.reshape(correct_side, (1, len(correct_side)))
        choice = np.reshape(choice, (1, len(choice)))
        correct_side[np.where(correct_side == -1)] = 2
        correct_side = np.abs(correct_side-3)
        performance = (choice == correct_side)
        evidence = stimulus[:, 1] - stimulus[:, 2]
        evidence = np.reshape(evidence, (1, len(evidence)))
        num_tr = 2000000
        start_point = performance.shape[1]-num_tr
        ev = evidence[:, start_point:start_point+num_tr]
        perf = performance[:, start_point:start_point+num_tr]
        ch = choice[:, start_point:start_point+num_tr]
        ch = np.reshape(ch, (ch.shape[1],))
        ev = np.reshape(ev, (ev.shape[1],))
        # BIAS CONDITIONED ON TRANSITION HISTORY (NUMBER OF REPETITIONS)
        conv_window = 8
        margin = 2
        transitions = get_transition_mat(ch, conv_window=conv_window)
        rand_perf = np.array(np.random.choice([0, 1])).reshape(1,)
        prev_perf = np.concatenate((rand_perf, perf[0, :-1]))
        mat_biases = np.empty((2, conv_window))
        labels = ['error', 'correct']
        ut.get_fig()
        for ind_perf in reversed(range(2)):
            plt.subplot(1, 2, int(not(ind_perf))+1)
            plt.title('after ' + labels[ind_perf])
            for ind_tr in range(margin, conv_window-margin+1):
                color = np.array((conv_window-ind_tr, 0, ind_tr))/conv_window
                mask = np.logical_and(transitions == ind_tr,
                                      prev_perf == ind_perf)
                assert np.sum(mask) > 20000
                popt, pcov = bias_calculation(ch, ev, mask)
                print('bias ' + str(ind_tr) + ' repeatitions after ' +
                      labels[ind_perf] + ':')
                print(popt[1])
                mat_biases[ind_perf, ind_tr] = popt[1]
                x = np.linspace(np.min(evidence),
                                np.max(evidence), 50)
                # get the y values for the fitting
                y = probit_lapse_rates(x, popt[0], popt[1], popt[2], popt[3])
                plt.plot(x, y, color=color,  label=str(ind_tr) +
                         ' b: ' + str(round(popt[1], 3)), lw=0.5)
            plt.xlim([-1.5, 1.5])
            plt.legend(loc="lower right")
            plot_dashed_lines(-np.max(evidence), np.max(evidence))
