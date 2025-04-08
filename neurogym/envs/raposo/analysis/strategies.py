import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# from raposo_task.analysis import chronometric as ch
# from raposo_task.analysis import psychometric as psy
import chronometric as ch
import psychometric as psy

IDENTIFIER = f'80onestepfull100'


def fit_psych_chron_data_all(data_path, frequencies, psy_fit_func, show_plot=True, pertubation_path='',average_modalities=False, lapse=False):
    psy_fit_params = psy.fit_psych_data_all(data_path, psy_fit_func, frequencies, False, pertubation_path,average_modalities=average_modalities)
    ch_mean_rt = ch.chron_plot_all(data_path, frequencies, False, pertubation_path,average_modalities=average_modalities)

    networks = os.listdir(data_path)
    networks = list(filter(lambda x: x.isnumeric(), networks))
    networks.sort(key=lambda x: int(x))  # sort the list of folders


   # print(data_path)
    if(average_modalities):
        lapse_gamma = np.zeros(len(networks))
        lapse_lambda = np.zeros(len(networks))
        slope = np.zeros(len(networks))
        bias = np.zeros(len(networks))
        rt = np.zeros(len(networks))
        for idx, net in enumerate(networks):
            slope[idx] = psy_fit_params[net]['mean_params'][0]
            bias[idx] = psy_fit_params[net]['mean_params'][1]
            rt[idx] = ch_mean_rt[net]['mean']
            if(lapse):
                lapse_gamma[idx] = psy_fit_params[net]['mean_params'][2]
                lapse_lambda[idx] = psy_fit_params[net]['mean_params'][3]


    else:
        lapse_gamma = np.zeros((len(networks), 3))
        lapse_lambda = np.zeros((len(networks), 3))
        slope = np.zeros((len(networks), 3))  # 3 corresponds to visual, auditory and multisensory
        bias = np.zeros((len(networks), 3))  # 3 corresponds to visual, auditory and multisensory
        rt = np.zeros((len(networks), 3))  # 3 corresponds to visual, auditory and multisensory
        for idx, net in enumerate(networks):

            # visual data
        # lapse_gamma[idx, 0] = psy_fit_params[net]['vis_params'][0]
            #lapse_lambda[idx, 0] = psy_fit_params[net]['vis_params'][1]
            slope[idx, 0] = psy_fit_params[net]['vis_params'][0]
            bias[idx, 0] = psy_fit_params[net]['vis_params'][1]
            rt[idx, 0] = ch_mean_rt[net]['v']

            # auditory data
        # lapse_gamma[idx, 1] = psy_fit_params[net]['aud_params'][0]
            #lapse_lambda[idx, 1] = psy_fit_params[net]['aud_params'][1]
            slope[idx, 1] = psy_fit_params[net]['aud_params'][0]
            bias[idx, 1] = psy_fit_params[net]['aud_params'][1]
            rt[idx, 1] = ch_mean_rt[net]['a']

            # multisensory data
        # lapse_gamma[idx, 2] = psy_fit_params[net]['ms_params'][0]
        # lapse_lambda[idx, 2] = psy_fit_params[net]['ms_params'][1]
            slope[idx, 2] = psy_fit_params[net]['ms_params'][0]
            bias[idx, 2] = psy_fit_params[net]['ms_params'][1]
            rt[idx, 2] = ch_mean_rt[net]['va']

    if show_plot is True:
        # plot histogram
        # slope
        f, ax = plt.subplots(1, 1)
        ax.hist(slope.ravel(), color='C0', density=True, rwidth=0.7)
        ax.tick_params(left=False, labelleft=False)
        # ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        ax.set_xlabel(r'Slope', fontsize=24)
        # format lines on all sides of the figure
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.savefig(os.path.join(f'../../stopping_figures/slope_hist_{IDENTIFIER}.png'))
        plt.show()
        plt.close()

        # bias
        f, ax = plt.subplots(1, 1)
        ax.hist(bias.ravel(), color='C0', density=True, rwidth=0.7)
        ax.tick_params(left=False, labelleft=False)
        # ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        ax.set_xlabel(r'Bias', fontsize=24)
        # format lines on all sides of the figure
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.savefig(os.path.join(f'../../stopping_figures/bias_hist_{IDENTIFIER}.png'))
        plt.show()
        plt.close()

        # RT
        f, ax = plt.subplots(1, 1)
        ax.hist(rt.ravel(), color='C0', density=True, rwidth=0.7)
        ax.tick_params(left=False, labelleft=False)
        # ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        ax.set_xlabel(r'Mean RT', fontsize=24)
        # format lines on all sides of the figure
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.savefig(os.path.join(f'../../stopping_figures/mean_rt_hist_{IDENTIFIER}.png'))
        plt.show()
        plt.close()

    extra_id = ''
    if lapse:
        extra_id = '_lapse'
    # save estimated stats
    if(average_modalities):
        if(lapse):
            np.save(data_path +  f'{pertubation_path}lapse_gamma_averaged', lapse_gamma)
            np.save(data_path +  f'{pertubation_path}lapse_lambda_averaged', lapse_lambda)

        np.save(data_path +  f'{pertubation_path}meant_rt_averaged{extra_id}', rt)
        np.save(data_path + f'{pertubation_path}slope_averaged{extra_id}', slope)
        np.save(data_path + f'{pertubation_path}bias_averaged{extra_id}', bias)
        #print('ye')
    else:
        np.save(data_path +  f'{pertubation_path}meant_rt{extra_id}', rt)
        np.save(data_path + f'{pertubation_path}slope{extra_id}', slope)
        np.save(data_path + f'{pertubation_path}bias{extra_id}', bias)
    #np.save(os.path.join(data_path, 'lapse_gamma'), lapse_gamma)
    #np.save(os.path.join(data_path, 'lapse_lambda'), lapse_lambda)
    if(lapse):
        return rt, slope, bias, lapse_gamma, lapse_lambda
    return rt, slope, bias


def est_quad(slope, mean_rt, slope_mid_point, rt_mid_point):
    if slope < slope_mid_point and mean_rt < rt_mid_point:
        return 0
    if slope > slope_mid_point and mean_rt < rt_mid_point:
        return 1
    if slope < slope_mid_point and mean_rt > rt_mid_point:
        return 2
    if slope > slope_mid_point and mean_rt > rt_mid_point:
        return 3


def est_net_per_quad(data_path, show_plot=True):
    net_per_quad = np.zeros((4, 3))  # 4 quadrants and 3 modalities

    # load mean-rt and slope data
    mean_rt = np.load(os.path.join(data_path, 'meant_rt.npy'))
    slope = np.load(os.path.join(data_path, 'slope.npy'))
    num_networks = mean_rt.shape[0]

    scaling_factor = 2 / 0.5
    slope_mid_point = (np.max(slope) + np.min(slope)) / 2 #25
    rt_mid_point = (np.max(mean_rt) + np.min(mean_rt)) / 2 #350 / scaling_factor
    for net in range(num_networks):
        # visual data
        quad = est_quad(slope[net, 0], mean_rt[net, 0], slope_mid_point, rt_mid_point)
        net_per_quad[quad, 0] += 1

        # auditory data
        quad = est_quad(slope[net, 1], mean_rt[net, 1], slope_mid_point, rt_mid_point)
        net_per_quad[quad, 1] += 1

        # multisensory data
        quad = est_quad(slope[net, 2], mean_rt[net, 2], slope_mid_point, rt_mid_point)
        net_per_quad[quad, 2] += 1

    if show_plot is True:
        quad_sum = np.sum(net_per_quad, axis=1)
        norm_quad_count = net_per_quad / quad_sum[:, None]

        f = plt.figure()
        for i in range(4):  # for four quadrants
            x_pos = np.arange(3) + (i * 3) + i
            plt.bar(x_pos, norm_quad_count[i, :], color=['C0', 'C1', 'C2'])

        ax = f.axes[0]
        # set fontsize for ticks
        ax.tick_params(labelsize=16)
        # tick labels
        labels = ['V', 'A', 'MS', ''] * 4
        labels = labels[:-1]
        plt.xticks(np.arange(15), labels)
        # fix axes limits
        ax.set_ylim([0, 1])
        # remove lines on the top and right
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        # add axes labels for quadrants
        transform = ax.get_xaxis_transform()
        # Q1
        ax.add_line(Line2D([-0.5, 2.5],
                           [-0.09] * 2,
                           transform=transform,
                           color="k", clip_on=False))
        ax.text(1.0, -0.11, 'Q1', transform=transform, ha="center", va="top", fontsize=16)
        # Q2
        ax.add_line(Line2D([3.5, 6.5],
                           [-0.09] * 2,
                           transform=transform,
                           color="k", clip_on=False))
        ax.text(5.0, -0.11, 'Q2', transform=transform, ha="center", va="top", fontsize=16)
        # Q3
        ax.add_line(Line2D([7.5, 10.5],
                           [-0.09] * 2,
                           transform=transform,
                           color="k", clip_on=False))
        ax.text(9.0, -0.11, 'Q3', transform=transform, ha="center", va="top", fontsize=16)
        # Q4
        ax.add_line(Line2D([11.5, 14.5],
                           [-0.09] * 2,
                           transform=transform,
                           color="k", clip_on=False))
        ax.text(13.0, -0.11, 'Q4', transform=transform, ha="center", va="top", fontsize=16)

        ax.set_ylabel('Normalized Count', fontsize=18)

        plt.tight_layout()
        plt.savefig(f'../../stopping_figures/net_per_quad_{IDENTIFIER}.png')
        plt.show()
        plt.close()


def begin():
    data_path = 'E:\ThesisWierda\ms-rnn\\'
    # task related data
    frequencies = np.arange(9, 17, 1)
    # fit psychometric data for all networks
    fit_psych_chron_data_all(data_path, frequencies, psy.sigmoid)

    # estimate the number of networks in each quadrant
    est_net_per_quad(data_path)


if __name__ == '__main__':
    begin()
