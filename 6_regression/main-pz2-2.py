import multiprocessing
import time
import pandas as pd
import scipy.io as sio

from os import sep
from scipy import stats
from itertools import product
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from main_function import relevant_information, sturcture_dataset, set_record, set_record2, set_writer, check_path, \
    nested_cv

from main_run import run
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
feature = ['wpli', ]
# ['merge', 'all', 'DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
band = ['DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
cluster_merge = True  # [True, False]

model_name_select = ['RVR',]
model_select = [RVR(kernel='linear', n_iter=10000),]
param_grid_select = [{},]

if __name__ == '__main__':
    multiprocess = False
    time_start = time.time()  # 记录开始时间

    data_merge = False  # [True, False]
    condition = ['EC', 'EO']
    repetition = 10
    outer_cv = 10
    inner_cv = 10

    ROIData_path = '../ROIData/' + SCZ + sep
    cluster_result_path = '../cluster_result/' + SCZ + sep
    information = SCZ + '_' + scale

    # lack_info_id, label, label_eng = relevant_information(information)
    lack_info_id = [1023, 1040, 2028, 2034, 3009, 3010, 3013]
    label = ['阴性分', '一般躯体症状', 'panss总分']
    label_eng = ['Negative', 'General', 'Panss']

    for model_name, model, param_grid in zip(model_name_select, model_select, param_grid_select):
        print(model_name, model, param_grid)
        if cluster_merge:
            result_path = './regre_result/' + SCZ + sep + scale + '_' + feature[0] + '_merge' + sep + model_name + sep
        else:
            result_path = './regre_result/' + SCZ + sep + scale + '_' + feature[0] + sep + model_name + sep
        check_path(result_path)

        for f in feature:
            writer, writer_record = set_writer(result_path, f)

            for la, title in zip(label, label_eng):
                writer.writerow(['', '', '', ''])
                writer.writerow([title, '', '', ''])
                writer_record.writerow(['', ])
                writer_record.writerow([title, ])

                for c in condition:
                    writer.writerow([c, '', '', ''])
                    writer_record.writerow([c, ])

                    for b in band:
                        writer.writerow([b, '', '', ''])
                        writer.writerow(['', 'r', 'p', 'mse'])
                        writer_record.writerow([b, ])

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
                            if not multiprocess:
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
                            else:
                                x0, y0 = sturcture_dataset(data, df_info, df_cluster, lack_info_id, b, la, data_merge,
                                                           0)
                                x1, y1 = sturcture_dataset(data, df_info, df_cluster, lack_info_id, b, la, data_merge,
                                                           1)
                                x, y = sturcture_dataset(data, df_info, df_cluster, lack_info_id, b, la, data_merge, 2)

                                # 对同一被试的所有维度特征进行zscore归一化
                                x0 = stats.zscore(x0, 1, ddof=1)
                                x1 = stats.zscore(x1, 1, ddof=1)
                                x = stats.zscore(x, 1, ddof=1)

                                from multiprocessing import Process, Queue, Pool, cpu_count

                                queue0 = Queue()
                                queue1 = Queue()
                                queue2 = Queue()

                                # cpu_num = cpu_count()
                                # p = Pool(cpu_num)
                                # process_names = [(x0, y0, model, param_grid, rep, outer_cv, inner_cv, False,
                                #                   record['s0'], False),
                                #                  (x1, y1, model, param_grid, rep, outer_cv, inner_cv, False,
                                #                   record['s1'], False),
                                #                  (x, y, model, param_grid, rep, outer_cv, inner_cv, False, record['s'],
                                #                   False)]
                                # for i in range(3):
                                #     res = p.apply(nested_cv, args=(process_names[i]))  # apply的结果就是func的返回值,同步提交
                                #     res = p.apply_async(nested_cv, args=(process_names[i]))  # apply_sync的结果就是异步获取func的返回值

                                # process_names = [(x0, y0, model, param_grid, rep, outer_cv, inner_cv, False,
                                #                   record['s0'], False, multiprocess, queue0),
                                #                  (x1, y1, model, param_grid, rep, outer_cv, inner_cv, False,
                                #                   record['s1'], False, multiprocess, queue1),
                                #                  (x, y, model, param_grid, rep, outer_cv, inner_cv, False, record['s'],
                                #                   False, multiprocess, queue2)]
                                # pool = Pool(processes=cpu_num)
                                # pool.map(nested_cv, process_names)
                                # pool.terminate()
                                # pool.close()  # 关闭进程池，不再接受新的进程
                                # pool.join()  # 主进程阻塞等待子进程的退出

                                # pool1 = Process(target=nested_cv, args=(
                                #     x0, y0, model, param_grid, rep, outer_cv, inner_cv, False, record['s0'], False,
                                #     multiprocess, queue0))
                                # pool2 = Process(target=nested_cv, args=(
                                #     x1, y1, model, param_grid, rep, outer_cv, inner_cv, False, record['s1'], False,
                                #     multiprocess, queue1))
                                # pool3 = Process(target=nested_cv, args=(
                                #     x, y, model, param_grid, rep, outer_cv, inner_cv, False, record['s'], False,
                                #     multiprocess, queue2))
                                # pool1.start()
                                # pool2.start()
                                # pool3.start()
                                # pool1.join()
                                # pool2.join()
                                # pool3.join()
                                # pool1.terminate()
                                # pool2.terminate()
                                # pool3.terminate()

                                return0 = queue0.get()
                                return1 = queue1.get()
                                return2 = queue2.get()
                                y_test = [return0[0], return1[0], return2[0]]
                                y_predict = [return0[1], return1[1], return2[1]]

                                for cluster, subtype in enumerate(['s0', 's1', 's']):
                                    r, p = pearsonr(y_test[cluster], y_predict[cluster])
                                    mse = mean_squared_error(y_test[cluster], y_predict[cluster])
                                    record_rep[cluster + 1, 6 * rep + 1] = mse
                                    record_rep[cluster + 1, 6 * rep + 2] = r
                                    record_rep[cluster + 1, 6 * rep + 3] = p
                                    y_test_cv[subtype].extend(y_test[cluster])
                                    y_predict_cv[subtype].extend(y_predict[cluster])

                        writer_record.writerows(record_rep)
                        for subtype in ['s0', 's1', 's']:
                            rr, pp = pearsonr(y_test_cv[subtype], y_predict_cv[subtype])
                            mse_all = mean_squared_error(y_test_cv[subtype], y_predict_cv[subtype])

                            writer.writerow([subtype, rr, pp, mse_all])
                            writer_record.writerows(record[subtype])

    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(time_sum)
