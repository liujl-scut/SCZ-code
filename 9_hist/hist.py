import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from os import sep
from collections import Counter

matplotlib.rcParams['font.family'] = 'STSong'
"""
对数据的临床量表（Panss，rbans）进行直方图统计，并保存绘制的直方图
"""

SCZ = 'lishui'
info_path = './' + SCZ + sep + 'patient_info_corr.csv'
panss = ['阳性分', '阴性分', '一般躯体症状', 'panss总分']
rbans = ['rbans维度1', 'rbans维度2', 'rbans维度3', 'rbans维度4', 'rbans维度5', 'rbans总分', '换算后']

lack_info_id = [1001, 1008, 1009, 1013, 1021, 1025, 2003,
                2006, 2012, 2013, 2017, 2039, 3007, 3008,
                3014, 3017, 3024, 3025, 3026, 4002, 4014,
                2021]
label = rbans
df = pd.read_csv(info_path)
if label == rbans:
    for i in lack_info_id:
        df = df.drop(df[df['id'] == i].index)

for la in label:
    c = Counter(df[la].to_list())
    values = c.keys()
    frequencies = c.values()

    plt.figure(figsize=(100, 20), dpi=80)
    plt.bar(values, frequencies)
    plt.xticks(range(int(min(values)), int(max(values)) + 1))

    plt.xlabel("临床量表分数")
    plt.ylabel("频数")
    plt.title(SCZ + '_' + la)

    plt.savefig(la + '.png')
    # plt.show()
    # plt.close()
