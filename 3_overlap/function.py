import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def sub_id(df):
    c1 = df.loc[df['cluster'] == 0]['subject'].to_list()
    c2 = df.loc[df['cluster'] == 1]['subject'].to_list()
    return c1, c2


def diff_sub_grid(a, b):
    diff_index = []
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            diff_index.append(i)
    return diff_index


def diff(a, b):
    diff1 = list(set(a) - set(b))
    diff2 = list(set(b) - set(a))
    len_diff = len(diff1) + len(diff2)
    return len_diff


def chi_square(e0, e1, o0, o1):
    exp = [len(e0), len(e1)]
    obs = [len(o0), len(o1)]
    chi2 = stats.chisquare(obs, exp)
    return chi2[0], chi2[1]


def define_cluster_label(a):
    c0_len = len(a.loc[a['cluster'] == 0.0]['cluster'])
    c1_len = len(a.loc[a['cluster'] == 1.0]['cluster'])
    if c0_len > c1_len:
        temp = 2
        a.loc[a['cluster'] == 0.0, ['cluster']] = temp
        a.loc[a['cluster'] == 1.0, ['cluster']] = 0
        a.loc[a['cluster'] == temp, ['cluster']] = 1
    return a


def overall_split(o, s):
    a0, a1 = sub_id(o)
    b0, b1 = sub_id(s)
    same00 = len(set(a0) & set(b0))
    same11 = len(set(a1) & set(b1))
    same01 = len(set(a0) & set(b1))
    same10 = len(set(a1) & set(b0))
    if same00 + same11 < same01 + same10:
        temp = 2
        s.loc[s['cluster'] == 0.0, ['cluster']] = temp
        s.loc[s['cluster'] == 1.0, ['cluster']] = 0
        s.loc[s['cluster'] == temp, ['cluster']] = 1
        return s
    else:
        return s


def sub_visual(oa, s1, s2, p):
    plt.figure(figsize=(10, 10))
    # plt.grid(axis="x")

    ax = plt.gca()
    ax.grid(axis='x', which='minor')
    grid_lines1 = diff_sub_grid(oa['cluster'], s1['cluster'])
    grid_lines2 = diff_sub_grid(oa['cluster'], s2['cluster'])
    grid_lines3 = diff_sub_grid(s1['cluster'], s2['cluster'])
    grid_lines = sorted(list(set(grid_lines1 + grid_lines2 + grid_lines3)))
    for line in grid_lines:
        ax.vlines(line, -0.2, 5, colors='k', linestyles='--', linewidth=0.4)

    plt.plot(oa['sub_index'], oa['cluster'], 'ro--', linewidth=0.5, label='overall')
    plt.plot(s1['sub_index'], s1['cluster'] + 2, 'go--', linewidth=0.5, label='half1')
    plt.plot(s2['sub_index'], s2['cluster'] + 4, 'bo--', linewidth=0.5, label='half2')

    plt.legend(loc="upper right")
    plt.xlabel('subject ID')
    plt.ylabel('label')
    plt.xticks(np.arange(0, len(oa['sub_index']), 1))

    # change x internal size
    plt.gca().margins(x=0.02)
    plt.gcf().canvas.draw()
    maxsize = 30
    m = 0.2  # inch margin
    s = maxsize / plt.gcf().dpi * len(oa['sub_index']) + 2 * m
    margin = m / plt.gcf().get_size_inches()[0]
    plt.gcf().subplots_adjust(left=margin, right=1. - margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])

    plt.savefig(p + 'sub_visual.png')
    plt.close()


def write_csv(csv_path, csv_list):
    file = open(csv_path, 'w', newline='')
    writer = csv.writer(file)
    for i in csv_list:
        writer.writerow(i)
    file.close()