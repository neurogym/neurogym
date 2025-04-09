import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, stats

THRESHOLD = 0.2 #decision threshold
IDENTIFIER = f'reportplot' #suffix plots

def  psych_analysis(data_path, th, frequencies):
    # load test trials and network response

    net_out = np.load(os.path.join(data_path, 'test_output.npy'))
    with open(os.path.join(data_path, 'test_trials.pkl'), 'rb') as f:
        trials = pickle.load(f)

    # difference between response of output neurons
    out_diff = net_out[:, trials['phases']['stimulus'], 1] - net_out[:, trials['phases']['stimulus'], 0]

    # time step when network made the decision
    decision_time = np.argmax(np.abs(out_diff) > th, axis=1)

    #Alternative
    # DECISION_THRESHOLD = 0.6
    # decision_time = np.minimum(
    #     np.argmax(np.abs(net_out[:, trials['phases']['stimulus'], 1]) > DECISION_THRESHOLD, axis=1),
    #     np.argmax(np.abs(net_out[:, trials['phases']['stimulus'], 0]) > DECISION_THRESHOLD, axis=1)
    #     )
    # discard trials where decision step is 0. A decision step of 0 means network made a choice before any stimulus is
    # presented

    analysed_trials = np.nonzero(decision_time != 0)[0]
    # predicted choice
    choice = (out_diff[analysed_trials, decision_time[analysed_trials]] > 0).astype(np.int_)


    # condition-wise analysis
    modality = ['v', 'a', 'va']
    psych_results = {}
    for m in modality:
        for f in frequencies:
            # trials associated with this condition
            cond_trials = np.nonzero(np.logical_and(trials['modality'][analysed_trials] == m,
                                                    trials['freq'][analysed_trials] == f))[0]

            if cond_trials.size != 0:
                n_cond_trials = cond_trials.size
                # percentage of trials where frequency is predicted as greater than boundary
                percent_higher = np.sum(choice[cond_trials] == 1) / n_cond_trials

                psych_cond_key = f'{m}-{f}'
                psych_results[psych_cond_key] = {}
                psych_results[psych_cond_key]['modality'] = m
                psych_results[psych_cond_key]['frequency'] = f
                psych_results[psych_cond_key]['trials'] = analysed_trials[cond_trials]
                psych_results[psych_cond_key]['choice'] = choice[cond_trials]
                psych_results[psych_cond_key]['accuracy'] = 100 * percent_higher


    n_analyzed_trials = analysed_trials.size
    if n_analyzed_trials > 0:
        accuracy = 100 * np.sum(trials['choice'][analysed_trials] == choice) / n_analyzed_trials
    else:
        accuracy = None
    return psych_results, n_analyzed_trials, accuracy

def  psych_analysis_songlike(data_path, th, frequencies):
    # load test trials and network response

    net_out = np.load(os.path.join(data_path, 'test_output.npy'))
    with open(os.path.join(data_path, 'test_trials.pkl'), 'rb') as f:
        trials = pickle.load(f)

    # difference between response of output neurons
    out_diff = net_out[:, trials['phases']['stimulus'], 1] - net_out[:, trials['phases']['stimulus'], 0]

    # time step when network made the decision
    decision_time = np.argmax(np.abs(out_diff) > th, axis=1)

    #Alternative
    DECISION_THRESHOLD = 0.6 #whichever output variable reaches this first

    decision_time_higher = np.argmax(np.abs(net_out[:, trials['phases']['stimulus'], 1]) > DECISION_THRESHOLD, axis=1)
    decision_time_lower = np.argmax(np.abs(net_out[:, trials['phases']['stimulus'], 0]) > DECISION_THRESHOLD, axis=1)

    decision_time_higher = decision_time_higher.astype(int)
    decision_time_lower = decision_time_lower.astype(int)
    decision_time = np.zeros(decision_time_higher.size)
    for i in range(decision_time_higher.size):
        if decision_time_higher[i] == 0 and decision_time_lower[i] == 0:
            decision_time[i] = 0
            continue
        if decision_time_higher[i] == 0:
            decision_time[i] = decision_time_lower[i]
            continue
        if decision_time_lower[i] == 0:
             decision_time[i] = decision_time_higher[i]
        else:
            decision_time[i] = np.min([decision_time_higher[i], decision_time_lower[i]])
    decision_time = decision_time.astype(int)

    analysed_trials = np.nonzero(decision_time != 0)[0]


    #analysed_trials = np.nonzero(decision_time != 0)[0]
    # predicted choice
    choice = (out_diff[analysed_trials, decision_time[analysed_trials]] > 0).astype(np.int_)


    # condition-wise analysis
    modality = ['v', 'a', 'va']
    psych_results = {}
    for m in modality:
        for f in frequencies:
            # trials associated with this condition
            cond_trials = np.nonzero(np.logical_and(trials['modality'][analysed_trials] == m,
                                                    trials['freq'][analysed_trials] == f))[0]

            if cond_trials.size != 0:
                n_cond_trials = cond_trials.size
                # percentage of trials where frequency is predicted as greater than boundary
                percent_higher = np.sum(choice[cond_trials] == 1) / n_cond_trials

                psych_cond_key = f'{m}-{f}'
                psych_results[psych_cond_key] = {}
                psych_results[psych_cond_key]['modality'] = m
                psych_results[psych_cond_key]['frequency'] = f
                psych_results[psych_cond_key]['trials'] = analysed_trials[cond_trials]
                psych_results[psych_cond_key]['choice'] = choice[cond_trials]
                psych_results[psych_cond_key]['accuracy'] = 100 * percent_higher


    n_analyzed_trials = analysed_trials.size
    if n_analyzed_trials > 0:
        accuracy = 100 * np.sum(trials['choice'][analysed_trials] == choice) / n_analyzed_trials
    else:
        accuracy = None
    return psych_results, n_analyzed_trials, accuracy

def psych_analysis_v2(data_path, th, frequencies):
    '''
    Function for psychometric curves as used in the paper
    '''
    net_out = np.load(os.path.join(data_path, 'test_output.npy'))
    with open(os.path.join(data_path, 'test_trials.pkl'), 'rb') as f:
        trials = pickle.load(f)

    # difference between response of output neurons
    out_diff = net_out[:, trials['phases']['stimulus'], 1] - net_out[:, trials['phases']['stimulus'], 0]

    # time step when network made the decision
    decision_time = np.argmax(np.abs(out_diff) > th, axis=1)

    out_diff_onset_stimulus = net_out[:, trials['phases']['stimulus'][0], 1] - net_out[:, trials['phases']['stimulus'][0], 0]


    #Difference in output is less than threshold at the start
    analysed_trials_valid_start = np.nonzero(np.abs(out_diff_onset_stimulus) <= th)[0]

    #A decision is made
    analysed_trials_choice_made = np.nonzero(np.sum(np.abs(out_diff) > th, axis=1) != 0)[0]

    #Both a valid start and a choice made
    analysed_trials_good_start_choice_made = np.intersect1d(analysed_trials_valid_start, analysed_trials_choice_made)


    # predicted choice
    choice = (out_diff[analysed_trials_good_start_choice_made, decision_time[analysed_trials_good_start_choice_made]] > 0).astype(np.int_)


    # condition-wise analysis
    modality = ['v', 'a', 'va']
    psych_results = {}
    for m in modality:
        for f in frequencies:
            # trials associated with this condition, but discard type 1 (choice made from start)
            cond_trials = np.nonzero(np.logical_and(trials['modality'][analysed_trials_valid_start] == m,
                                                    trials['freq'][analysed_trials_valid_start] == f))[0]

            #Trials with choices in the condition (type 2 and 3)
            cond_trials_with_choice = np.nonzero(np.logical_and(trials['modality'][analysed_trials_good_start_choice_made] == m,
                                                    trials['freq'][analysed_trials_good_start_choice_made] == f))[0]

            #print(cond_trials_with_choice.size)
            if cond_trials_with_choice.size != 0:
                #n_cond_trials = cond_trials.size
                n_cond_trials = cond_trials_with_choice.size
                # percentage of trials where frequency is predicted as greater than boundary
                percent_higher = np.sum(choice[cond_trials_with_choice] == 1) / n_cond_trials #!NEW HIGHER OUT OF ALL CHOICES, NOT JUST ANALYZED

                psych_cond_key = f'{m}-{f}'

                psych_results[psych_cond_key] = {}
                psych_results[psych_cond_key]['modality'] = m
                psych_results[psych_cond_key]['frequency'] = f
                psych_results[psych_cond_key]['trials'] = analysed_trials_good_start_choice_made[cond_trials_with_choice]
                psych_results[psych_cond_key]['choice'] = choice[cond_trials_with_choice]
                psych_results[psych_cond_key]['accuracy'] = 100 * percent_higher


    #n_analyzed_trials = analysed_trials.size
    n_analyzed_trials = len(analysed_trials_valid_start)
    #print(len(choice))

    if n_analyzed_trials > 0:
        accuracy_over_choices = 100 * np.sum(trials['choice'][analysed_trials_good_start_choice_made] == choice) / n_analyzed_trials #decision_time.size #!Alternatively, n_analyzed_trials
        accuracy_over_all = 100 * np.sum(trials['choice'][analysed_trials_good_start_choice_made] == choice) / decision_time.size
        accuracy = accuracy_over_choices
    else:
        accuracy_over_choices = None
        accuracy_over_all = None
        accuracy = None

    return psych_results, n_analyzed_trials, accuracy


def  psych_analysis_v3(data_path, th, frequencies):

    net_out = np.load(os.path.join(data_path, 'test_output.npy'))
    with open(os.path.join(data_path, 'test_trials.pkl'), 'rb') as f:
        trials = pickle.load(f)

    # difference between response of output neurons
    out_diff = net_out[:, trials['phases']['stimulus'], 1] - net_out[:, trials['phases']['stimulus'], 0]
    out_diff_onset_stimulus = net_out[:, trials['phases']['stimulus'][0], 1] - net_out[:, trials['phases']['stimulus'][0], 0]

    # time step when network made the decision
    decision_time = np.argmax(np.abs(out_diff) > th, axis=1)

    #if no decision made or decision made from start: put decision time at last time step
    decision_time[decision_time == 0] = len(trials['phases']['stimulus']) - 1


    #Difference in output is less than threshold at the start, ignore trial
    analysed_trials_valid_start = np.nonzero(np.abs(out_diff_onset_stimulus) <= th)[0]


    analysed_trials = np.nonzero(decision_time != 0)[0]

    # predicted choice
    choice = (out_diff[analysed_trials_valid_start, decision_time[analysed_trials_valid_start]] > 0).astype(np.int_)


    # condition-wise analysis
    modality = ['v', 'a', 'va']
    psych_results = {}
    for m in modality:
        for f in frequencies:
            # trials associated with this condition, but discard type 1 (choice made from start)
            cond_trials = np.nonzero(np.logical_and(trials['modality'][analysed_trials_valid_start] == m,
                                                    trials['freq'][analysed_trials_valid_start] == f))[0]

            if cond_trials.size != 0:
                n_cond_trials = cond_trials.size
                # percentage of trials where frequency is predicted as greater than boundary
                percent_higher = np.sum(choice[cond_trials] == 1) / n_cond_trials

                psych_cond_key = f'{m}-{f}'

                psych_results[psych_cond_key] = {}
                psych_results[psych_cond_key]['modality'] = m
                psych_results[psych_cond_key]['frequency'] = f
                psych_results[psych_cond_key]['trials'] = analysed_trials_valid_start
                psych_results[psych_cond_key]['choice'] = choice[n_cond_trials]
                psych_results[psych_cond_key]['accuracy'] = 100 * percent_higher


    n_analyzed_trials = len(analysed_trials_valid_start)

    if n_analyzed_trials > 0:
        accuracy_over_choices = 100 * np.sum(trials['choice'][analysed_trials_valid_start] == choice) / n_analyzed_trials #decision_time.size #!Alternatively, n_analyzed_trials
        accuracy_over_all = 100 * np.sum(trials['choice'][analysed_trials_valid_start] == choice) / decision_time.size
        accuracy = accuracy_over_choices
    else:
        accuracy_over_choices = None
        accuracy_over_all = None
        accuracy = None

    return psych_results, n_analyzed_trials, accuracy


def psych_plot(psych_results, frequencies, th, data_path, show_plots):
    frequencies = frequencies.tolist()

    vis_perf = np.zeros(len(frequencies))
    aud_perf = np.zeros(len(frequencies))
    ms_perf = np.zeros(len(frequencies))

    for key in psych_results:
        value = psych_results[key]
        modality = value['modality']
        idx = frequencies.index(value['frequency'])
        if modality == 'v':
            vis_perf[idx] = value['accuracy'] #! percentage of higher
        elif modality == 'a':
            aud_perf[idx] = value['accuracy']
        elif modality == 'va':
            ms_perf[idx] = value['accuracy']



    if show_plots is True:
        # generate psychometric plot
        f, ax = plt.subplots(1, 1)
        ax.plot(frequencies, vis_perf, 'o-', label='visual')
        ax.plot(frequencies, aud_perf, 'o-', label='auditory')
        ax.plot(frequencies, ms_perf, 'o-', label='multisensory')

        ax.set_ylabel('Percentage of Left Choices', fontsize=18)
        ax.set_xlabel(r'Frequency', fontsize=18)
        ax.tick_params(labelsize=14)
        ax.set_title(f'Threshold = {th}', fontsize=20)
        plt.tight_layout()
        plt.legend(prop={'size': 14})

        fig_suffix = str(th).replace('.', '')
        plt.savefig(os.path.join(data_path, f'psych_plot_{fig_suffix}_{IDENTIFIER}.png'))
        # plt.show()
        # plt.close()

    return vis_perf, aud_perf, ms_perf


def psych_plot_all(data_path, frequencies):
    networks = os.listdir(data_path)
    networks = list(filter(lambda x: x.isnumeric(), networks))
    networks.sort(key=lambda x: int(x))  # sort the list of folders

    vis_perf = {}
    aud_perf = {}
    ms_perf = {}
    for net in networks:
        psych_results, th = sel_threshold(os.path.join(data_path, net), frequencies)

        net_vis_perf, net_aud_perf, net_ms_perf = psych_plot(psych_results, frequencies, th,
                                                             os.path.join(data_path, net), False)

        vis_perf[net] = net_vis_perf
        aud_perf[net] = net_aud_perf
        ms_perf[net] = net_ms_perf

        print(f'psych_plot_all: Processed network {net}')

    # visual data
    f, ax = plt.subplots(1, 1)
    alphas = np.linspace(0.1, 1, len(networks))
    for idx, net in enumerate(networks):
        ax.plot(frequencies, vis_perf[net], 'o-', color='C0', alpha=alphas[idx])

    # set axes labels
    ax.set_ylabel('Frequency > Central Frequency (%)', fontsize=16)
    ax.set_xlabel(r'Frequency', fontsize=16)
    ax.tick_params(labelsize=14)
    # set figure title
    f.suptitle(f'Visual Only', fontsize=16)
    # format lines on all sides of the figure
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    # save the figure
    plt.tight_layout()

    print(f'current directory: {os.getcwd()}')
    plt.savefig(os.path.join(f'stopping_figures/vis_psych_plot_all_{IDENTIFIER}.png'))
    plt.show()
    plt.close()

    # auditory data
    f, ax = plt.subplots(1, 1)
    for idx, net in enumerate(networks):
        ax.plot(frequencies, aud_perf[net], 'o-', color='C1', alpha=alphas[idx])

    ax.set_ylabel('Frequency > Central Frequency (%)', fontsize=16)
    ax.set_xlabel(r'Frequency', fontsize=16)
    ax.tick_params(labelsize=14)
    f.suptitle(f'Auditory Only', fontsize=16)
    # format lines on all sides of the figure
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    # save the figure
    plt.tight_layout()

    print(f'current directory: {os.getcwd()}')
    plt.savefig(os.path.join(f'stopping_figures/aud_psych_plot_all__{IDENTIFIER}.png'))
    plt.show()
    plt.close()

    # multisensory data
    f, ax = plt.subplots(1, 1)
    for idx, net in enumerate(networks):
        ax.plot(frequencies, ms_perf[net], 'o-', color='C2', alpha=alphas[idx])

    ax.set_ylabel('Frequency > Central Frequency (%)', fontsize=16)
    ax.set_xlabel(r'Frequency', fontsize=16)
    ax.tick_params(labelsize=14)
    f.suptitle(f'Multisensory', fontsize=16)
    # format lines on all sides of the figure
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    # save the figure
    plt.tight_layout()

    print(f'current directory: {os.getcwd()}')
    plt.savefig(os.path.join(f'stopping_figures/ms_psych_plot_all__{IDENTIFIER}.png'))
    #plt.show()
    plt.close()


def sigmoid(x, a, b):
    return 1 / (1 + np.exp(- (a * (x - b))))


def fit_psych_data(data_path, fit_function, frequencies, sel_th=True, show_plots=True,pertubation_path='', average_modalities=False):
    # parameters for analysis
    th = 0.2
    center_freq = np.mean(frequencies)
    freq_step_plot = 0.01  # used for plotting fitted function

    data_path = data_path + pertubation_path

    # We need to specify which network!
    netw = "10"
    # psychometric analysis
    if sel_th is True:
        psych_results, th = sel_threshold(os.path.join(data_path,netw), frequencies)

    else:
        psych_results, _, _ = psych_analysis_v2(os.path.join(data_path, netw), th, frequencies)

    #print(psych_results)
    vis_perf, aud_perf, ms_perf = psych_plot(psych_results, frequencies, th, data_path, show_plots)
    vis_perf = vis_perf / 100
    aud_perf = aud_perf / 100
    ms_perf = ms_perf / 100

    if(average_modalities):
        mean_perf = np.mean([vis_perf, aud_perf, ms_perf], axis=0)


    # fit function to the visual psychometric data
    fit_params = {}
    delta_freq = np.array(frequencies) - center_freq

    if(average_modalities):
        mean_params, mean_covar = optimize.curve_fit(fit_function, delta_freq, mean_perf, method='trf', maxfev=10000)
        fit_params['mean_params'] = mean_params
        fit_params['mean_covar'] = mean_covar
        return fit_params, th

    vis_params, vis_covar = optimize.curve_fit(fit_function, delta_freq, vis_perf, method='trf', maxfev=10000)
    fit_params['vis_params'] = vis_params
    fit_params['vis_covar'] = vis_covar
    # fit sigmoid function to the auditory psychometric plots
    aud_params, aud_covar = optimize.curve_fit(fit_function, delta_freq, aud_perf, method='trf', maxfev=10000)
    fit_params['aud_params'] = aud_params
    fit_params['aud_covar'] = aud_covar
    # fit sigmoid function to the multisensory psychometric plots
    ms_params, ms_covar = optimize.curve_fit(fit_function, delta_freq, ms_perf, method='trf', maxfev=10000)
    fit_params['ms_params'] = ms_params
    fit_params['ms_covar'] = ms_covar

    if show_plots is True:
        # expand range of frequency for smoothening the plot
        fit_delta_freq = np.arange(delta_freq[0], delta_freq[-1] + freq_step_plot, freq_step_plot)
        # plot fitted function for checking
        f = plt.figure(figsize=(6, 4))
        plt.scatter(delta_freq, vis_perf, label='visual data')
        plt.plot(fit_delta_freq, fit_function(fit_delta_freq, vis_params[0], vis_params[1]), label='visual fit')
        plt.scatter(delta_freq, aud_perf, label='auditory data')
        plt.plot(fit_delta_freq, fit_function(fit_delta_freq, aud_params[0], aud_params[1]), label='auditory fit')
        plt.scatter(delta_freq, ms_perf, label='multisensory data')
        plt.plot(fit_delta_freq, fit_function(fit_delta_freq, ms_params[0], ms_params[1]), label='multisensory fit')

        ax = f.axes[0]
        ax.set_ylabel('Proportion of Left Choices', fontsize=16)
        ax.set_xlabel(r'Frequency - Central Frequency', fontsize=16)
        ax.tick_params(labelsize=14)

        ax.set_title(f'Threshold = {th}', fontsize=16)
        plt.legend(prop={'size': 12})
        plt.tight_layout()

        fig_suffix = str(th).replace('.', '')
        plt.savefig(os.path.join(data_path, f'psych_fit_{fig_suffix}_{IDENTIFIER}.png'))
        plt.show()
        plt.close()

    return fit_params, th


def sel_threshold(data_path, frequencies):
    '''
    This function now passes a fixed threshold of value THRESHOLD, but could easily be changed to
    pass an array of threshold of which to judge the psychometric function
    '''
    min_threshold = 0.1
    max_threshold = 0.2
    threshold_step = 0.01
    threshold_range = np.arange(min_threshold, max_threshold + threshold_step, threshold_step).tolist()

    th_cond_met = False
    #print(data_path)
    for th in [max_threshold]:
        th = round(th, 2)  # numpy looses precision at times. This is a fail-safe.
        psych_results, n_analyzed_trials, accuracy =  psych_analysis_v2(data_path, THRESHOLD, frequencies)


    return psych_results, th


def fit_psych_data_all(data_path, fit_function, frequencies, show_plot=True, pertubation_path='', average_modalities=False):
    """
    :param data_path: directory where data for all networks is present
    :param fit_function:
    :param frequencies:
    :return:
    """

    center_freq = np.mean(frequencies)
    freq_step_plot = 0.01  # used for plotting fitted function
    networks = os.listdir(data_path)
    networks = list(filter(lambda x: x.isnumeric(), networks))
    networks.sort(key=lambda x: int(x))  # sort the list of folders

    fit_params = {}
    th = {}
    for net in networks:
        net_fit_params, net_th = fit_psych_data(os.path.join(data_path, net), fit_function, frequencies,sel_th=True,
                                                show_plots=False, pertubation_path=pertubation_path, average_modalities=average_modalities)
        fit_params[net] = net_fit_params
        th[net] = net_th

        print(f'fit_psych_data_all: Fiting complete for network {net}')

    if show_plot is True and not average_modalities :
        delta_freq = np.array(frequencies) - center_freq
        fit_delta_freq = np.arange(delta_freq[0], delta_freq[-1] + freq_step_plot, freq_step_plot)
        # accuracy of the networks estimated using fitted functions for the three types of stimuli
        vis_accuracy = np.zeros((len(networks), fit_delta_freq.size))
        aud_accuracy = np.zeros((len(networks), fit_delta_freq.size))
        ms_accuracy = np.zeros((len(networks), fit_delta_freq.size))
        for idx, net in enumerate(networks):
            # visual stimuli
            vis_accuracy[idx, :] = fit_function(fit_delta_freq, fit_params[net]['vis_params'][0],
                                                fit_params[net]['vis_params'][1])
            # auditory stimuli
            aud_accuracy[idx, :] = fit_function(fit_delta_freq, fit_params[net]['aud_params'][0],
                                                fit_params[net]['aud_params'][1])
            # visual stimuli
            ms_accuracy[idx, :] = fit_function(fit_delta_freq, fit_params[net]['ms_params'][0],
                                               fit_params[net]['ms_params'][1])

        # plots with SEM
        f, ax = plt.subplots(1, 1)

        # visual stimuli
        vis_mean = np.mean(vis_accuracy, axis=0)
        sem_plus = vis_mean + stats.sem(vis_accuracy)
        sem_minus = vis_mean - stats.sem(vis_accuracy)
        ax.fill_between(fit_delta_freq + center_freq, sem_plus, sem_minus, alpha=0.5)
        ax.plot(fit_delta_freq + center_freq, vis_mean, label='visual')

        # auditory stimuli
        aud_mean = np.mean(aud_accuracy, axis=0)
        sem_plus = aud_mean + stats.sem(aud_accuracy)
        sem_minus = aud_mean - stats.sem(aud_accuracy)
        ax.fill_between(fit_delta_freq + center_freq, sem_plus, sem_minus, alpha=0.5)
        ax.plot(fit_delta_freq + center_freq, aud_mean, label='auditory')

        # multisensory stimuli
        ms_mean = np.mean(ms_accuracy, axis=0)
        sem_plus = ms_mean + stats.sem(ms_accuracy)
        sem_minus = ms_mean - stats.sem(ms_accuracy)
        ax.fill_between(fit_delta_freq + center_freq, sem_plus, sem_minus, alpha=0.5)
        ax.plot(fit_delta_freq + center_freq, ms_mean, label='multisensory')

        #ax.set_xticklabels(np.arange(9, 17, 1))
        ax.set_yticklabels(np.arange(-20, 101, 20))
        ax.set_ylabel('Proportion of Left Choices', fontsize=16)  # 'left' = Frequency lower than central frequency
        ax.set_xlabel(r'Frequency', fontsize=16)
        ax.tick_params(labelsize=14)

        # format lines on all sides of the figure
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)

        plt.legend(prop={'size': 12})
        plt.tight_layout()

        print(f'current directory: {os.getcwd()}')

        plt.savefig(os.path.join(f'stopping_figures/psych_fit_sem_plot_{IDENTIFIER}.png'))
        plt.show()
        plt.close()

    return fit_params


def begin():

    data_path = '/Users/lexotto/Documents_Mac/Stage/UVA/Code/BehavioralVariabilityRNN-main/data'
    # task related data
    frequencies = np.arange(9, 17, 1)

    # analyze and plot psychometric curves
    # Specify the network that needs to be analysed
    netw = "10"
    psych_results, th = sel_threshold(os.path.join(data_path, netw), frequencies)  # select threshold automatically
    psych_plot(psych_results, frequencies, th, data_path, True)

    # plot psychometric data for all networks
    psych_plot_all(data_path, frequencies)

    # fit psychometric data for a single network
    fit_psych_data(data_path, sigmoid, frequencies, show_plots=True)

    # fit psychometric data for all networks
    fit_psych_data_all(data_path, sigmoid, frequencies)


if __name__ == '__main__':
    begin()
