import pandas as pd
import scipy.io as sio

from os import sep
from scipy import stats
from itertools import product
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from function import relevant_information, sturcture_dataset, nested_cv, set_record, set_record2, set_writer, check_path

from skrvm import RVR
from sklearn.svm import SVR
from sklearn_rvm import EMRVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, SGDRegressor, ElasticNet

SCZ = 'lishui'  # ['lishui', 'ningbo']
scale = 'panss'  # ['panss', 'rbans']

# ['pec', 'wpli', 'icoh']
feature = ['pec', ]
# ['merge', 'all', 'DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
band = ['ALPHA', 'BETA']
data_merge = False  # [True, False]
cluster_merge = False  # [True, False]

condition = ['EC', 'EO']
repetition = 10
outer_cv = 10
inner_cv = 10

# model_name_select = ['SVR', 'Lasso', 'RVR']
# model_select = [SVR(kernel='linear'), Lasso(max_iter=10000), RVR(kernel='linear')]
# param_grid_select = [{'C': [0.0001, 0.001, 0.01, 0.1, 1.0]},
#                      {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]},
#                      {}]
# model_name_select = ['LinearRegression', 'KNeighborsRegressor', 'DecisionTreeRegressor', 'RandomForestRegressor',
#                      'MLPRegressor', 'KernelRidge', 'SGDRegressor', 'ElasticNet']
# model_select = [LinearRegression(), KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor(),
#                 MLPRegressor(max_iter=10000), KernelRidge(), SGDRegressor(max_iter=10000, tol=1e-3),
#                 ElasticNet(random_state=0, alpha=0.15, l1_ratio=0.9)]
# param_grid_select = [{},
#                      {'n_neighbors': [3, 4, 5, 6, 7, 8]},
#                      {},
#                      {'n_estimators': [30, 50, 100]},
#                      {'learning_rate_init': [0.01, 0.1]},
#                      {'alpha': [0.001, 0.01, 0.1]},
#                      {},
#                      {}]
model_name_select = ['SVR']
model_select = [SVR(kernel='linear')]
param_grid_select = [{'C': [0.0001, 0.001, 0.01, 0.1, ]}]

ROIData_path = '../ROIData/' + SCZ + sep
cluster_result_path = '../cluster_result/' + SCZ + sep
information = SCZ + '_' + scale
lack_info_id, label, label_eng = relevant_information(information)

for model_name, model, param_grid in zip(model_name_select, model_select, param_grid_select):
    print(model_name, model, param_grid)
    result_path = './regre_result/' + SCZ + sep + scale + '_' + feature[0] + sep + model_name + sep
    check_path(result_path)

    for f in feature:
        writer, writer_record = set_writer(result_path, f)

        for la, title in zip(label, label_eng):
            writer.writerow(['', '', '', ''])
            writer.writerow([title, '', '', ''])
            writer_record.writerow(['',])
            writer_record.writerow([title,])

            for c in condition:
                writer.writerow([c, '', '', ''])
                writer_record.writerow([c,])

                for b in band:
                    writer.writerow([b, '', '', ''])
                    writer.writerow(['', 'r', 'p', 'mse'])
                    writer_record.writerow([b,])

                    data_path = ROIData_path + c + '_' + f + '.mat'
                    info_path = ROIData_path + 'patient_info_corr.csv'
                    if cluster_merge:
                        cluster_path = cluster_result_path + f + '/merge/2/overall.csv'
                    else:
                        cluster_path = cluster_result_path + f + sep + b + '/2/overall.csv'

                    data = sio.loadmat(data_path)
                    df_info = pd.read_csv(info_path)
                    df_cluster = pd.read_csv(cluster_path, encoding="utf-8")

                    y_test_cv = {'s0': [], 's1': [], 's': []}
                    y_predict_cv = {'s0': [], 's1': [], 's': []}

                    # 记录交叉验证训练过程中的值
                    record = set_record(outer_cv, repetition)
                    record_rep = set_record2(repetition)

                    for rep in range(repetition):
                        for cluster, subtype in enumerate(['s0', 's1', 's']):  # 选择subtype
                            x, y = sturcture_dataset(data, df_info, df_cluster, lack_info_id, b, la, data_merge,
                                                     cluster)
                            x = stats.zscore(x, 1, ddof=1)  # 对同一被试的所有维度特征进行zscore归一化
                            y_test, y_predict = nested_cv(x, y, model, param_grid, rep,
                                                          outer_cv, inner_cv,
                                                          True, record[subtype])

                            r, p = pearsonr(y_test, y_predict)
                            mse = mean_squared_error(y_test, y_predict)
                            record_rep[cluster + 1, 6 * rep + 1] = mse
                            record_rep[cluster + 1, 6 * rep + 2] = r
                            record_rep[cluster + 1, 6 * rep + 3] = p

                            y_test_cv[subtype].extend(y_test)
                            y_predict_cv[subtype].extend(y_predict)

                    writer_record.writerows(record_rep)
                    for subtype in ['s0', 's1', 's']:
                        rr, pp = pearsonr(y_test_cv[subtype], y_predict_cv[subtype])
                        mse_all = mean_squared_error(y_test_cv[subtype], y_predict_cv[subtype])

                        writer.writerow([subtype, rr, pp, mse_all])
                        writer_record.writerows(record[subtype])
