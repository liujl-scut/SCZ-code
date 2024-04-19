import pandas as pd
from os import sep
from func import make_radar_data, radar, lack_id, norm

SCZ = 'ningbo'
scale = 'panss'
info = SCZ + '_' + scale
save_path = './radar_char/'

lack_id = lack_id(info)

for feature in ['pec', 'wpli']:
    for band in ['merge', 'all', 'DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']:
        cluster_path = '../cluster_result/' + SCZ + sep + feature + sep + band + '/2/overall.csv'
        info_path = '../ROIData/' + SCZ + sep + 'patient_info_corr.csv'
        df_cluster = pd.read_csv(cluster_path, encoding="utf-8")
        df_info = pd.read_csv(info_path)

        y0, y1, y = make_radar_data(df_cluster, df_info, scale, lack_id)
        y0, y1, y = norm(y0), norm(y1), norm(y)
        radar(y0.copy(), y1.copy(), y.copy(), scale,
              save_path + sep + scale + '_' + feature + '_' + band + '.jpg')
        # radar(y0[0:5].copy(), y1[0:5].copy(), y[0:5].copy(), scale,
        #       save_path + sep + scale + '_' + feature + '_' + band + '.jpg')
