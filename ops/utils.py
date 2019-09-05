import numpy as np
import os
import difflib


def look_for_folder(main_folder='priors/', exp=''):
    """
    looks for a given folder and returns it.
    If it cannot find it, returns possible candidates
    """
    data_path = ''
    possibilities = []
    for root, dirs, files in os.walk(main_folder):
        ind = root.rfind('/')
        possibilities.append(root[ind+1:])
        if root[ind+1:] == exp:
            data_path = root
            break

    if data_path == '':
        candidates = difflib.get_close_matches(exp, possibilities,
                                               n=1, cutoff=0.)
        print(exp + ' NOT FOUND IN ' + main_folder)
        if len(candidates) > 0:
            print('possible candidates:')
            print(candidates)

    return data_path


def list_str(l):
    """
    list to str
    """
    if isinstance(l, (list, tuple, np.ndarray)):
        nice_string = str(l[0])
        for ind_el in range(1, len(l)):
            nice_string += '_'+str(l[ind_el])
    else:
        nice_string = str(l)

    return nice_string


def num2str(num):
    """
    pass big number to thousands
    """
    string = ''
    while num/1000 >= 1:
        string += 'K'
        num = num/1000
    string = str(int(num)) + string
    return string


def rm_lines():
    """
    remove all lines from the current axis
    """
    import matplotlib.pyplot as plt
    ax = plt.gca()
    ax.clear()


def plot_trials_start(trials, minimo, maximo, num_steps, color='k'):
    """
    plot dashed lines that indicate the end of the current trial
    and the start of the next one
    """
    import matplotlib.pyplot as plt
    trials = np.nonzero(trials)[0] - 0.5
    cond = np.logical_and(trials >= 0, trials <= num_steps)
    trials = trials[np.where(cond)]
    for ind_tr in range(len(trials)):
        plt.plot([trials[ind_tr], trials[ind_tr]], [minimo, maximo],
                 '--'+color, lw=1)
    plt.xlim(0-0.5, num_steps-0.5)


def folder_name(gamma=0.8, up_net=5, trial_dur=10,
                rep_prob=(.2, .8), exp_dur=10**6,
                rewards=(-0.1, 0.0, 1.0, -1.0),
                block_dur=200, num_units=32,
                stim_ev=0.5, network='ugru', learning_rate=10e-3,
                instance=0, main_folder=''):
    """
    build the name of the folder where data are saved
    """
    return main_folder + '/td_' + str(trial_dur) + '_rp_' +\
        str(list_str(rep_prob)) + '_r_' +\
        str(list_str(rewards)) + '_bd_' + str(block_dur) +\
        '_ev_' + str(stim_ev) + '_g_' + str(gamma) + '_lr_' +\
        str(learning_rate) + '_nu_' + str(num_units) + '_un_' + str(up_net) +\
        '_net_' + str(network) + '_ed_' + str(num2str(exp_dur)) +\
        '_' + str(instance) + '/'


def color_axis(ax, color='r'):
    ax.spines['bottom'].set_color(color)
    ax.spines['top'].set_color(color)
    ax.spines['right'].set_color(color)
    ax.spines['left'].set_color(color)


def get_fig(display_mode=None, font=6, figsize=(8, 8)):
    import matplotlib
    if display_mode is not None:
        if display_mode:
            matplotlib.use('Qt5Agg')
        else:
            matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    left = 0.125  # the left side of the subplots of the figure
    right = 0.9  # the right side of the subplots of the figure
    bottom = 0.1  # the bottom of the subplots of the figure
    top = 0.9  # the top of the subplots of the figure
    wspace = 0.4  # width reserved for blank space between subplots
    hspace = 0.4  # height reserved for white space between subplots
    f = plt.figure(figsize=figsize, dpi=250)
    matplotlib.rcParams.update({'font.size': font, 'lines.linewidth': 0.5,
                                'axes.titlepad': 1, 'lines.markersize': 3})
    plt.subplots_adjust(left=left, bottom=bottom, right=right,
                        top=top, wspace=wspace, hspace=hspace)
    return f


def order_by_sufix(file_list):
    sfx = [int(x[x.rfind('_')+1:x.rfind('.')]) for x in file_list]
    sorted_list = [x for _, x in sorted(zip(sfx, file_list))]
    return sorted_list
