import os
import pandas as pd
import scipy.io as sio
from function import concat_images, plot_matrix_31x31, plot_ttest, convert_feature465_matrix31x31, ttest

'''
根据二聚类的结果，进行数据转换。
数据转换为将维度为465的特征数据，转换为31*31的矩阵，计算出同一类下的平均矩阵。
结果得到4张子图，分别为EC和EO两种情况的两类矩阵
'''
# SCZ = 'ningbo'
SCZ = 'lishui'
feature = ['pec', 'wpli', 'icoh']
bandnames = ['all', 'DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
base_path = 'D:/myTask/cluster/ROI/'
SAVE_QUALITY = 50  # 保存的图片的质量 可选0-100

for f in feature:
    EC_tmat = [''] * 6
    EC_diffmat = [''] * 6
    EO_tmat = [''] * 6
    EO_diffmat = [''] * 6

    EC_tmat_merge = [''] * 6
    EC_diffmat_merge = [''] * 6
    EO_tmat_merge = [''] * 6
    EO_diffmat_merge = [''] * 6

    for i, band in enumerate(bandnames):
        data_path_EC = base_path + SCZ + '/EC_' + f + '.mat'
        data_path_EO = base_path + SCZ + '/EO_' + f + '.mat'
        cluster_path = base_path + SCZ + os.sep + f + os.sep + band + '/2/overall.csv'
        save_path = base_path + SCZ + os.sep + f + os.sep
        cluster_path_merge = base_path + SCZ + os.sep + f + '/merge/2/overall.csv'
        save_path_merge = base_path + SCZ + os.sep + f + '/merge/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_path_merge):
            os.makedirs(save_path_merge)

        data_EC = sio.loadmat(data_path_EC)
        data_EO = sio.loadmat(data_path_EO)
        df = pd.read_csv(cluster_path, encoding="utf-8")
        df_merge = pd.read_csv(cluster_path_merge, encoding="utf-8")

        # ['all', 'DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
        mat_EC, max_EC, min_EC, EC_fea465 = convert_feature465_matrix31x31(data_EC, df,
                                                                           band)  # mat_EC:[(31, 31), (31, 31)]
        mat_EO, max_EO, min_EO, EO_fea465 = convert_feature465_matrix31x31(data_EO, df,
                                                                           band)  # mat_EO:[(31, 31), (31, 31)]
        max_value = max_EC if max_EC > max_EO else max_EO
        min_value = min_EC if min_EC < min_EO else min_EO
        EC_tmat[i], EC_diffmat[i] = ttest(EC_fea465)
        EO_tmat[i], EO_diffmat[i] = ttest(EO_fea465)
        plot_matrix_31x31(mat_EC, mat_EO, min_value, max_value, save_path, band)

        # merge
        mat_EC_merge, max_EC, min_EC, EC_fea465_merge = convert_feature465_matrix31x31(data_EC, df_merge,
                                                                                       band)  # mat_EC:[(31, 31), (31, 31)]
        mat_EO_merge, max_EO, min_EO, EO_fea465_merge = convert_feature465_matrix31x31(data_EO, df_merge,
                                                                                       band)  # mat_EO:[(31, 31), (31, 31)]
        max_value = max_EC if max_EC > max_EO else max_EO
        min_value = min_EC if min_EC < min_EO else min_EO
        EC_tmat_merge[i], EC_diffmat_merge[i] = ttest(EC_fea465_merge)
        EO_tmat_merge[i], EO_diffmat_merge[i] = ttest(EO_fea465_merge)
        plot_matrix_31x31(mat_EC_merge, mat_EO_merge, min_value, max_value, save_path_merge, band)

    plot_ttest(EC_tmat, save_path, 'tvalue_EC')
    plot_ttest(EO_tmat, save_path, 'tvalue_EO')
    plot_ttest(EC_tmat_merge, save_path_merge, 'tvalue_EC')
    plot_ttest(EO_tmat_merge, save_path_merge, 'tvalue_EO')

    # 图片拼接
    c = 3  # 指定拼接图片的列数
    r = 2  # 指定拼接图片的行数
    h = 2000  # 图片高度
    w = 2000  # 图片宽度
    PATH_LIST = ['all.jpg', 'DELTA.jpg', 'THETA.jpg', 'ALPHA.jpg', 'BETA.jpg', 'GAMMA.jpg']
    NAME = "1"  # 拼接出的图片保存的名字
    concat_images(PATH_LIST, NAME, save_path, c, r, h, w, SAVE_QUALITY)
    concat_images(PATH_LIST, NAME, save_path_merge, c, r, h, w, SAVE_QUALITY)

    c = 1  # 指定拼接图片的列数
    r = 2  # 指定拼接图片的行数
    h = 2000  # 图片高度
    w = 3000  # 图片宽度
    PATH_LIST = ['tvalue_EC.jpg', 'tvalue_EO.jpg']
    NAME = "2"  # 拼接出的图片保存的名字
    concat_images(PATH_LIST, NAME, save_path, c, r, h, w, SAVE_QUALITY)
    concat_images(PATH_LIST, NAME, save_path_merge, c, r, h, w, SAVE_QUALITY)

    # plotTtest(EC_diffmat, save_path, 'diff_EC')
    # plotTtest(EO_diffmat, save_path, 'diff_EO')
    # plotTtest(EC_diffmat_merge, save_path_merge, 'diff_EC')
    # plotTtest(EO_diffmat_merge, save_path_merge, 'diff_EO')
