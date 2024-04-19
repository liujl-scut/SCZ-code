import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi


def lack_id(info):
    if info == 'lishui_panss':
        lack_info_id = [1023, 1040, 2028, 2034, 3009, 3010, 3013]
    elif info == 'lishui_rbans':
        lack_info_id = [1023, 1040, 2028, 2034, 3009, 3010, 3013,
                        1001, 1008, 1009, 1013, 1021, 1025, 2003,
                        2006, 2012, 2013, 2017, 2039, 3007, 3008,
                        3014, 3017, 3024, 3025, 3026, 4002, 4014,
                        2021]
    else:
        lack_info_id = [5, 22, 24, 25, 27, 28, 36, 39, 45, 51, 91, 99, 102]
    return lack_info_id


def make_radar_data(df_cluster, df_info, scale, LACK_INFO_ID):
    y0 = []
    y1 = []
    y = []
    for class_ in range(3):
        if class_ == 2:
            sub_cluster = df_cluster['subject'].tolist()
        else:
            sub_cluster = df_cluster.loc[df_cluster['cluster'] == class_]['subject'].tolist()

        # Delete irrelevant id
        sub_cluster = np.setdiff1d(sub_cluster, LACK_INFO_ID)

        if scale == 'panss':
            p = []
            n = []
            g = []
            panss = []
            for idx in sub_cluster:
                p = np.append(p, df_info.loc[df_info['id'] == idx]['阳性分'].tolist()[0])
                n = np.append(n, df_info.loc[df_info['id'] == idx]['阴性分'].tolist()[0])
                g = np.append(g, df_info.loc[df_info['id'] == idx]['一般躯体症状'].tolist()[0])
                panss = np.append(panss, df_info.loc[df_info['id'] == idx]['panss总分'].tolist()[0])
            if class_ == 0:
                y0 = [p.mean(), n.mean(), g.mean(), panss.mean()]
            elif class_ == 1:
                y1 = [p.mean(), n.mean(), g.mean(), panss.mean()]
            else:
                y = [p.mean(), n.mean(), g.mean(), panss.mean()]
        else:
            r1 = []
            r2 = []
            r3 = []
            r4 = []
            r5 = []
            rbans = []
            r_convert = []
            for idx in sub_cluster:
                r1 = np.append(r1, df_info.loc[df_info['id'] == idx]['rbans维度1'].tolist()[0])
                r2 = np.append(r2, df_info.loc[df_info['id'] == idx]['rbans维度2'].tolist()[0])
                r3 = np.append(r3, df_info.loc[df_info['id'] == idx]['rbans维度3'].tolist()[0])
                r4 = np.append(r4, df_info.loc[df_info['id'] == idx]['rbans维度4'].tolist()[0])
                r5 = np.append(r5, df_info.loc[df_info['id'] == idx]['rbans维度5'].tolist()[0])
                rbans = np.append(rbans, df_info.loc[df_info['id'] == idx]['rbans总分'].tolist()[0])
                r_convert = np.append(r_convert, df_info.loc[df_info['id'] == idx]['换算后'].tolist()[0])
            if class_ == 0:
                y0 = [r1.mean(), r2.mean(), r3.mean(), r4.mean(), r5.mean(), rbans.mean(), r_convert.mean()]
            elif class_ == 1:
                y1 = [r1.mean(), r2.mean(), r3.mean(), r4.mean(), r5.mean(), rbans.mean(), r_convert.mean()]
            else:
                y = [r1.mean(), r2.mean(), r3.mean(), r4.mean(), r5.mean(), rbans.mean(), r_convert.mean()]
    return y0, y1, y


def radar(y0, y1, y, scale, path):
    # number of variable
    # 变量类别
    if scale == 'panss':
        categories = ['Positive', 'Negative', 'General', 'Panss']
        N = len(categories)  # 变量类别个数
    else:
        # categories = ['rbans_1', 'rbans_2', 'rbans_3', 'rbans_4', 'rbans_5']
        categories = ['rbans_1', 'rbans_2', 'rbans_3', 'rbans_4', 'rbans_5', 'rbans', 'rbans_convert']
        N = len(categories)  # 变量类别个数

    plt.figure(figsize=(30, 30))
    # 设置每个点的角度值
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    # 初始化极坐标网格
    ax = plt.subplot(111, polar=True)
    # If you want the first axis to be on top:
    # 设置角度偏移
    ax.set_theta_offset(pi / 2)
    # 设置顺时针还是逆时针，1或者-1
    ax.set_theta_direction(-1)
    # Draw one axe per variable + add labels labels yet
    # 设置x轴的标签
    plt.xticks(angles[:-1], categories)
    # Draw ylabels
    # 画标签
    ax.set_rlabel_position(0)
    # plt.yticks([10, 20, 30], ["10", "20", "30"], color="grey", size=7)
    # plt.ylim(0, 1)
    plt.yticks(color="grey", size=7)

    # 单独绘制每一组数据
    # subtype 0
    y0 += y0[:1]
    ax.plot(angles, y0, linewidth=1, linestyle='solid', label="subtype 0")
    ax.fill(angles, y0, 'b', alpha=0.1)

    # subtype 1
    y1 += y1[:1]
    ax.plot(angles, y1, linewidth=1, linestyle='solid', label="subtype 1")
    ax.fill(angles, y1, 'r', alpha=0.1)

    # all patients
    y += y[:1]
    ax.plot(angles, y, linewidth=1, linestyle='solid', label="all patients")
    ax.fill(angles, y, 'r', alpha=0.1)

    # Add legend
    # 添加图例
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.savefig(path)
    plt.close()


def norm(y, scale='panss'):
    if scale == 'panss':
        y_norm = [(y[0] - 7) / (49 - 7), (y[1] - 7) / (49 - 7), (y[2] - 16) / (112 - 16), (y[3] - 30) / 210]
    else:
        y_norm = y
    return y_norm
