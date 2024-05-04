import numpy as np
import pandas as pd

lishui_path = './lishui.xlsx'
ningbo_path = './ningbo.xlsx'

# 读取xls（绝对路径）
lishui = pd.read_excel(lishui_path, engine='openpyxl', sheet_name=None, header=None)
ningbo = pd.read_excel(ningbo_path, engine='openpyxl', sheet_name=None, header=None)
label = ['Positive', 'Negative', 'General', 'Panss']
condition = ['EC', 'EO']


for key in ningbo.keys():
    df = pd.DataFrame(data=None, columns=[''])
    lishui_sheet = lishui[key]
    ningbo_sheet = ningbo[key]

    for i in range(11):
        out_con = np.array([[lishui_sheet.iloc[0, i * 4 + 1], '', '', '', '', '', '', '']])
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

                        out_1_2 = np.concatenate((out1.values, np.ones((4, 1)) * np.nan), axis=1)
                        out_1_2 = np.concatenate((out_1_2, out2.values), axis=1)
                        out_1_2 = np.insert(out_1_2, 0, values=[condition[k]] + [''] * (out_1_2.shape[1] - 1), axis=0)
                        out_1_2 = np.insert(out_1_2, 0, values=[label[j]] + [''] * (out_1_2.shape[1] - 1), axis=0)
                        out_1_2 = np.concatenate((out_1_2, np.ones((6, 1)) * np.nan), axis=1)
                        out_1_2 = np.concatenate((out_1_2, np.ones((1, 8)) * np.nan), axis=0)

                        out_con = np.concatenate((out_con, out_1_2), axis=0)

        df = pd.concat([df, pd.DataFrame(out_con)], axis=1)

    df.to_csv('./' + key + '.csv', index=False, header=False)
