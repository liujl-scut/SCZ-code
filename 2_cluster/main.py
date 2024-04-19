import os
import numpy as np
import scipy.io as sio
from func1_sparse_cluster import cluster, check
from func2_compute_metric import compute_metrics

if __name__ == '__main__':
    seed = 0
    SCZ = 'lishui'
    class_range = range(2, 3)
    feature = ['pec', 'wpli']       # ['pec', 'wpli', 'icoh']
    split_half = ['overall', 'split1', 'split2']  # ['overall', 'split1', 'split2']
    bandnames = ['merge', 'DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA', 'all']
    metric = ['Stability', 'SSE', 'Silhouette', 'CalinskiHarabasz', 'GapStatistic']

    # 是否进行交叉验证实验
    validation = False
    validation_runs = 5
    validation_ratio = 0.9

    ROI_path = 'D:/SCZ/code/cluster/ROI/' + SCZ + os.sep
    result_path = 'D:/SCZ/code/cluster/result/' + SCZ + os.sep

    for f in feature:
        os.chdir(ROI_path)
        ROI_EC = sio.loadmat('EC_' + f + '.mat')
        ROI_EO = sio.loadmat('EO_' + f + '.mat')
        data_EC = ROI_EC['ROI']
        data_EO = ROI_EO['ROI']
        sub_number = list(np.squeeze(data_EC['sub_number'][0, 0]))
        number = len(sub_number)

        for split in split_half:
            EC = data_EC[split][0, 0]
            EO = data_EO[split][0, 0]

            for b in bandnames:
                save_path = result_path + f + os.sep + b
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                # 5个频段的EC和EO数据进行融合，然后对融合数据进行聚类实验
                # merge_EC_EO格式：
                # [DELTA_EC; THETA_EC; ALPHA_EC; BETA_EC; GAMMA_EC;
                # DELTA_EO; THETA_EO; ALPHA_EO; BETA_EO; GAMMA_EO;]
                if b == 'merge':
                    merge_EC = np.zeros([number, 1])
                    merge_EO = np.zeros([number, 1])
                    for band in ['DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']:
                        merge_EC = np.concatenate((merge_EC, EC[band][0, 0]), axis=1)
                        merge_EO = np.concatenate((merge_EO, EO[band][0, 0]), axis=1)
                    merge_EC = merge_EC[:, 1:]
                    merge_EO = merge_EO[:, 1:]
                    merge_EC_EO = np.concatenate((merge_EC, merge_EO), axis=1)
                    data_merge, idx_merge = check(merge_EC_EO, sub_number)     # 数据异常值筛选

                    # 聚类实验
                    cluster(data_merge, idx_merge, save_path, class_range, seed, split, validation, validation_runs, validation_ratio)

                    # 计算指标（代码有一些指标未完成）
                    # compute_metrics(data_merge, idx_merge, save_path, class_range, validation_runs, validation_ratio, seed)  # 计算指标（未完成）

                # 单一频段的EC和EO进行融合，然后进行聚类实验
                # merge格式:[b_EC, b_EO] (b in ['DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA'])
                else:
                    merge = np.concatenate((EC[b][0, 0], EO[b][0, 0]), axis=1)
                    data, idx = check(merge, sub_number)        # 数据异常值筛选

                    # 聚类实验
                    cluster(data, idx, save_path, class_range, seed, split, validation, validation_runs, validation_ratio)

                    # 计算指标（代码有一些指标未完成）
                    # compute_metrics(data, idx, save_path, class_range, validation_runs, validation_ratio, seed)
