import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def scores_plot(scores, scores_std, elbow_index, cluster_range, ylabel, title, filename, path, std_weight=1):
    os.chdir(path)
    font = {'family': 'Arial', 'weight': 'normal'}
    scores_mean = np.mean(scores, axis=1)
    # scores_std = np.std(scores, axis=0)
    fig = plt.figure(figsize=(5, 5), dpi=200)
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(cluster_range)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    plt.tick_params(labelsize=12, width=2)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.2)

    plt.xlabel('Number of clusters', fontdict=font, fontsize=16)
    plt.ylabel(ylabel, fontdict=font, fontsize=16)
    plt.title(title, fontdict=font, fontsize=20)
    plt.xlim((cluster_range[0] - 0.5, cluster_range[-1] + 0.5))
    # plt.plot(cluster_range, TSSE_mean, color='black', linestyle='-', marker='s')
    if np.isnan(scores_mean[0]):
        scores_mean = scores_mean[1:]
        scores_std = scores_std[1:]
        cluster_range = cluster_range[1:]
        elbow_index = elbow_index - 1

    ymin = min(scores_mean - std_weight * scores_std)
    ymax = max(scores_mean + std_weight * scores_std)
    plt.ylim((ymin - (ymax - ymin) * 0.2, ymax + (ymax - ymin) * 0.2))
    plt.errorbar(cluster_range, scores_mean, yerr=std_weight * scores_std, color='black', linestyle='-', marker='s',
                 ecolor='black',
                 capsize=3)
    plt.errorbar(cluster_range[elbow_index], scores_mean[elbow_index], yerr=std_weight * scores_std[elbow_index],
                 color='red',
                 linestyle='-', marker='s', ecolor='red', capsize=3)
    plt.savefig(filename)
    plt.savefig(filename[:-4] + '.svg')
    plt.close()


def scores_result_save(scores, scores_cluster, cluster_range, filename, path):
    os.chdir(path)
    f = open(filename, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    headline = ['Scores\\No.cluster']
    [headline.append(cluster_range[i]) for i in range(len(cluster_range))]
    csv_writer.writerow(headline)
    secondline = ['WholeCluster']
    [secondline.append(scores[i, 0]) for i in range(len(cluster_range))]
    csv_writer.writerow(secondline)

    for csv_index in range(scores_cluster.shape[1]):
        csv_line = [csv_index]
        [csv_line.append(scores_cluster[i, csv_index]) for i in range(len(cluster_range))]
        csv_writer.writerow(csv_line)
    f.close()


def SSE_elbow_rule(data):
    CPV = np.array([np.sum(data[:i + 1]) / np.sum(data) for i in range(data.shape[0])])
    elbow_index = np.min(np.argwhere(CPV >= 0.9))
    return elbow_index
