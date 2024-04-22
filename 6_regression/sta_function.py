import os
import csv
import numpy as np
import pandas as pd
from openpyxl import Workbook

condition = ['EC', 'EO']
band = ['all', 'DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
panss = ['Positive', 'Negative', 'General', 'Panss']
rbans = ['rbans_1', 'rbans_2', 'rbans_3', 'rbans_4', 'rbans_5', 'rbans', 'rbans_convert']


def write_sta(filepath, filename):
    result_path = os.path.join(filepath, filename)
    sta_path = os.path.join(filepath, filename[0:-4] + '_sta.csv')
    df = pd.read_csv(result_path, header=None)
    file = open(sta_path, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(file)

    for index, row in df.iterrows():
        if row[0] in panss or row[0] in rbans or row[0] in condition:
            csv_writer.writerow([row[0], '', ''])
        elif pd.isna(row[0]) and pd.isna(row[1]) and pd.isna(row[2]):
            csv_writer.writerow(['', '', ''])
        elif row[0] in band:
            if (float(df.iloc[index + 2][2]) < 0.06 and float(df.iloc[index + 2][1]) > 0) or (
                    float(df.iloc[index + 3][2]) < 0.06 and float(df.iloc[index + 3][1]) > 0):
                csv_writer.writerow([row[0], '', ''])
                csv_writer.writerow([df.iloc[index + 2][0], df.iloc[index + 2][1], df.iloc[index + 2][2]])
                csv_writer.writerow([df.iloc[index + 3][0], df.iloc[index + 3][1], df.iloc[index + 3][2]])
                csv_writer.writerow([df.iloc[index + 4][0], df.iloc[index + 4][1], df.iloc[index + 4][2]])


def result_statistics_1(path):
    for filepath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            csv_path = os.path.join(filepath, filename)
            if (not csv_path.endswith('_record.csv')) and (not csv_path.endswith('_sta.csv')) and (
                    not csv_path.endswith('_sta.xlsx')):
                write_sta(filepath, filename)


def result_statistics_2(path):
    for root, dirs, files in os.walk(path):
        # 选择第二层子目录
        if root.count(os.path.sep) == path.count(os.path.sep) + 1:
            df = pd.DataFrame(data=None, columns=[''])

            for d in dirs:
                p = os.path.join(root, d)
                for _, _, filenames in os.walk(p):
                    for f in filenames:
                        if f.endswith('_sta.csv'):
                            sta_path = os.path.join(p, f)
                            df_add = pd.read_csv(sta_path, header=None)
                            df_add = pd.DataFrame(np.insert(df_add.values, 0, values=[d, '', ''], axis=0))
                            df_add['4'] = None
                            df = pd.concat([df, df_add], axis=1)
                            df.to_csv(root + '/model_sta.csv', index=False, header=False)


def result_statistics_3(path):
    walker = os.walk(path)
    root, dirs, _ = next(walker)
    label_sta_path = root + '/label_sta.xlsx'

    wb = Workbook()
    wb.save(label_sta_path)

    for d in dirs:
        model_sta_path = os.path.join(root, d)
        model_sta_path = model_sta_path + '/model_sta.csv'
        df = pd.read_csv(model_sta_path, header=None)

        with pd.ExcelWriter(label_sta_path, mode='a', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=d, header=False, index=False)
