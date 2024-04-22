import os
import csv
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_writer(result_path, f):
    filename1 = result_path + f + '.csv'
    filename2 = result_path + f + '_record.csv'
    file1 = open(filename1, 'w', encoding='utf-8', newline='')
    file2 = open(filename2, 'w', encoding='utf-8', newline='')
    writer1 = csv.writer(file1)
    writer2 = csv.writer(file2)
    return writer1, writer2


def set_record(outer_cv, repetition):
    record = {'s0': np.zeros(shape=(1 + outer_cv, 6 * repetition)).astype(np.str_),
              's1': np.zeros(shape=(1 + outer_cv, 6 * repetition)).astype(np.str_),
              's': np.zeros(shape=(1 + outer_cv, 6 * repetition)).astype(np.str_)}
    for subtype in ['s0', 's1', 's']:
        for i in range(repetition):
            record[subtype][0, 0 + i * 6] = subtype + ' outer_loop'
            record[subtype][0, 1 + i * 6] = 'mse'
            record[subtype][0, 2 + i * 6] = 'r'
            record[subtype][0, 3 + i * 6] = 'pvalue'
            record[subtype][0, 4 + i * 6] = 'model'
            record[subtype][0, 5 + i * 6] = 'inner_loop mse'
    return record


def set_record2(repetition):
    record_rep = np.zeros(shape=(4, 6 * repetition)).astype(np.str_)
    for i in range(repetition):
        record_rep[0, 0 + i * 6] = 'rep' + str(i)
        record_rep[1, 0 + i * 6] = 's0'
        record_rep[2, 0 + i * 6] = 's1'
        record_rep[3, 0 + i * 6] = 's'
        record_rep[0, 1 + i * 6] = 'mse'
        record_rep[0, 2 + i * 6] = 'r'
        record_rep[0, 3 + i * 6] = 'pvalue'
    return record_rep


def my_pearsonr_score(y_true, y_pred):
    r, _ = pearsonr(y_true, y_pred)
    return r


def relevant_information(info):
    if info == 'lishui_panss':
        lack_info_id = [1023, 1040, 2028, 2034, 3009, 3010, 3013]
        label = ['阳性分', '阴性分', '一般躯体症状', 'panss总分']
        label_eng = ['Positive', 'Negative', 'General', 'Panss']
    elif info == 'lishui_rbans':
        lack_info_id = [1023, 1040, 2028, 2034, 3009, 3010, 3013,
                        1001, 1008, 1009, 1013, 1021, 1025, 2003,
                        2006, 2012, 2013, 2017, 2039, 3007, 3008,
                        3014, 3017, 3024, 3025, 3026, 4002, 4014,
                        2021]
        label = ['rbans维度1', 'rbans维度2', 'rbans维度3', 'rbans维度4', 'rbans维度5', 'rbans总分', '换算后']
        label_eng = ['rbans_1', 'rbans_2', 'rbans_3', 'rbans_4', 'rbans_5', 'rbans', 'rbans_convert']
    else:
        lack_info_id = [5, 22, 24, 25, 27, 28, 36, 39, 45, 51, 91, 99, 102]
        label = ['阳性分', '阴性分', '一般躯体症状', 'panss总分']
        label_eng = ['Positive', 'Negative', 'General', 'Panss']
    return lack_info_id, label, label_eng


def sturcture_dataset(ROI, df_info, df_cluster, lack_info_id, band, label, data_merge, C):
    """
        根据聚类结果的id来构造训练测试数据集以及标签
        Args:
            :param ROI: 输入的ROI数据
            :param df_info: patient_info相关信息
            :param df_cluster: k_means对数据进行（二）聚类的结果
            :param lack_info_id: patient_info缺少的被试者id号
            :param band: 使用数据的频段
            :param label: 使用patient_info的标签
            :param data_merge: 是否对data数据进行全频段融合操作
            :param C: C==0或者1时，构造subtype0或者subtype1数据集；C==2时，构造所有被试者数据集

        Returns:
            x:训练测试数据集
            y:标签
    """
    data = ROI['ROI'][0, 0]['overall'][0, 0]
    sub_number = ROI['ROI']['sub_number'][0, 0]  # (N, 1)
    if not data_merge:
        data = data[band]  # (N, 465)
    else:
        data = [data[b] for b in ['DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']]
        data = np.concatenate(data, axis=1)  # (N, 465*5)
    if C == 2:
        subtype_id = df_cluster['subject'].tolist()  # (N,)
    else:
        subtype_id = df_cluster.loc[df_cluster['cluster'] == C]['subject'].tolist()  # (N0,)
    subtype_id = np.setdiff1d(subtype_id, lack_info_id)  # Delete irrelevant id
    index = [np.argwhere(sub_number == i)[0, 0] for i in subtype_id]
    x = data[index,]  # (N0, 465)
    y = [df_info.loc[df_info['id'] == i][label].tolist()[0] for i in subtype_id]  # (N0)
    y = np.array(y)
    return x, y


def nested_cv(X, Y, model, param_grid, rep, outer_fold=10, inner_fold=10, record=True, s_record=None,
              show_process=False):
    inner_cv = KFold(n_splits=inner_fold, shuffle=True, random_state=100 + rep)
    outer_cv = KFold(n_splits=outer_fold, shuffle=True, random_state=200 + rep)
    outer_loop = outer_cv.split(X, Y)

    y_predict_outercv = np.array([])
    y_test_outercv = np.array([])

    for k, (train_index, test_index) in enumerate(outer_loop):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        score = make_scorer(mean_squared_error, greater_is_better=False)
        regr = GridSearchCV(estimator=model, param_grid=param_grid, scoring=score, cv=inner_cv, n_jobs=-1)
        regr.fit(x_train, y_train)

        best_clf = regr.best_estimator_
        y_predict = best_clf.predict(x_test)

        y_predict_outercv = np.concatenate((y_predict_outercv, y_predict), axis=0)
        y_test_outercv = np.concatenate((y_test_outercv, y_test), axis=0)

        if show_process:
            means = regr.cv_results_['mean_test_score']
            params = regr.cv_results_['params']
            for mean, param in zip(means, params):
                print("%f  with:   %r" % (mean, param))
            print("Best: %f using %s" % (regr.best_score_, regr.best_params_))
            print()

        if record:
            print(y_test, y_predict)
            mse = mean_squared_error(y_test, y_predict)
            r, pvalue = pearsonr(y_test, y_predict)
            s_record[k + 1, 0 + rep * 6] = k
            s_record[k + 1, 1 + rep * 6] = mse
            s_record[k + 1, 2 + rep * 6] = r
            s_record[k + 1, 3 + rep * 6] = pvalue
            s_record[k + 1, 4 + rep * 6] = str(regr.best_estimator_)
            s_record[k + 1, 5 + rep * 6] = - regr.best_score_

    return y_test_outercv, y_predict_outercv
