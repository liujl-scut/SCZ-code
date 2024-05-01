import csv
import numpy as np
import pandas as pd

# 读取xls（绝对路径）
lishui = pd.read_excel('./regre_result/lishui.xlsx', engine='openpyxl', sheet_name=None, header=None)
ningbo = pd.read_excel('./regre_result/ningbo.xlsx', engine='openpyxl', sheet_name=None, header=None)
label = ['Positive', 'Negative', 'General', 'Panss']
condition = ['EC', 'EO']

filename = './regre_result/compare.csv'
file = open(filename, 'w', encoding='utf-8', newline='')
writer = csv.writer(file)

for key in ningbo.keys():
    writer.writerow([key, ])
    lishui_sheet = lishui[key]
    ningbo_sheet = ningbo[key]

    for i in range(11):
        writer.writerow([lishui_sheet.iloc[0, i * 4 + 1], '', ''])
        sheet1 = lishui_sheet.iloc[:, i * 4 + 1:i * 4 + 4]
        sheet2 = ningbo_sheet.iloc[:, i * 4 + 1:i * 4 + 4]

        P_1 = sheet1.where(sheet1 == 'Positive').stack().index[0][0]
        N_1 = sheet1.where(sheet1 == 'Negative').stack().index[0][0]
        G_1 = sheet1.where(sheet1 == 'General').stack().index[0][0]
        Panss_1 = sheet1.where(sheet1 == 'Panss').stack().index[0][0]
        End_1 = sheet1.index.stop
        i_1 = [P_1, N_1, G_1, Panss_1, End_1]

        P_2 = sheet2.where(sheet2 == 'Positive').stack().index[0][0]
        N_2 = sheet2.where(sheet2 == 'Negative').stack().index[0][0]
        G_2 = sheet2.where(sheet2 == 'General').stack().index[0][0]
        Panss_2 = sheet2.where(sheet2 == 'Panss').stack().index[0][0]
        End_2 = sheet2.index.stop
        i_2 = [P_2, N_2, G_2, Panss_2, End_2]

        for j in range(4):
            block1 = sheet1.iloc[i_1[j]:i_1[j + 1], :]
            block2 = sheet2.iloc[i_2[j]:i_2[j + 1], :]

            EC_1 = block1.where(block1 == 'EC').stack().index[0][0]
            EO_1 = block1.where(block1 == 'EO').stack().index[0][0]
            Eend_1 = block1.index.stop
            j_1 = [EC_1, EO_1, Eend_1]

            EC_2 = block2.where(block2 == 'EC').stack().index[0][0]
            EO_2 = block2.where(block2 == 'EO').stack().index[0][0]
            Eend_2 = block2.index.stop
            j_2 = [EC_2, EO_2, Eend_2]

            for k in range(2):
                compare_1 = sheet1.iloc[j_1[k]:j_1[k + 1], :]
                compare_2 = sheet2.iloc[j_2[k]:j_2[k + 1], :]

                for f in ['all', 'DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']:
                    if f in compare_1.values and f in compare_2.values:
                        index1 = compare_1.where(compare_1 == f).stack().index[0][0]
                        index2 = compare_2.where(compare_2 == f).stack().index[0][0]
                        out1 = sheet1.iloc[index1:index1 + 4, :]
                        out2 = sheet2.iloc[index2:index2 + 4, :]

                        writer.writerow([label[j], ])
                        writer.writerow([condition[k], ])

                        for m in range(4):
                            w = np.append(out1.values[m], '')
                            w = np.append(w, out2.values[m])
                            writer.writerows([w])
                        writer.writerow(['', '', ''])
