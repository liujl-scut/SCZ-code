import os
import pandas as pd

"""
根据raw EEG data的名称，来检查并重写确定lishui的patient_info表格中的id号
"""

folder = './lishui/EC/'
sub_id = []
name = []

for f in os.listdir(folder):
    sub_id.append(f[0:4])
    name.append(f[5:-29])

df = pd.read_csv('./patient_info.csv')
df['姓名拼音'] = df['姓名拼音'].str.lower()
df2 = pd.DataFrame(columns=df.columns)

for i in range(len(name)):
    if name[i] in df['姓名拼音'].unique():
        mask = df['姓名拼音'] == name[i]
        row = df[mask].iloc[0]
        row['id'] = sub_id[i]
        df2 = df2.append(row)
    else:
        print(sub_id[i], name[i])
df2.to_csv('output.csv', index=False)
