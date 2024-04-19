import os
import random
import pickle
import numpy as np
from func1_sparse_cluster import write_csv
from sub_function import scores_plot, scores_result_save


def check_train_val(train_center, val_center, val_label):
    cluster_num = train_center.shape[0]
    label_val_swap = np.full(val_label.shape, np.nan)
    distances = np.empty((cluster_num, cluster_num))

    for true in range(cluster_num):
        for val in range(cluster_num):
            distances[true, val] = np.linalg.norm(train_center[true, :] - val_center[val, :])

    for c_num in range(cluster_num):
        index = np.unravel_index(distances.argmin(), distances.shape)
        label_val_swap[val_label == index[1]] = index[0]
        distances[index[0], :] = np.inf
        distances[:, index[1]] = np.inf

    return label_val_swap


def SSE_compute(data, targets, centers):
    SSE = []
    for label in set(targets):
        SSE.append(np.sum((data[targets == label, :] - centers[int(label), :]) ** 2))
    SSE = np.sum(SSE)  # 计算总的簇内离差平方和
    # SSE = np.sum(SSE) / len(SSE)    # 这里是旧代码计算SSE的公式，感觉存在问题，mark一下
    return SSE


def metric_Stability(train_centers, train_targets, val_centers, val_targets,
                     subjects, save_path, dataset_name, cluster_range, runs, n_sample, number_k):
    accuracy = np.zeros((number_k, runs))
    for k_index, k in enumerate(cluster_range):
        centers = train_centers[k_index]

        for r in range(runs):
            val_centers_r = val_centers[r][k_index]
            val_targets_r = val_targets[r, k_index, :]
            val_targets_r_swap = check_train_val(centers, val_centers_r, val_targets_r)
            accuracy[k_index, r] = 100 * np.sum(train_targets[k_index, :] == val_targets_r_swap) / n_sample

        filename = 'No_' + str(k) + '_val_result.csv'
        cluster_path = save_path + os.sep + str(k)
        write_csv(subjects, np.squeeze(val_targets[:, k_index, :]).T, filename, cluster_path)

    accuracy_std = np.std(accuracy, axis=1)
    title = dataset_name
    ylabel = 'Stability (%)'
    fig_filename = 'Stability.jpg'
    std_weight = 1
    if cluster_range[0] == 1:
        elbow_index = np.argmax(np.mean(accuracy, axis=1)[1:]) + 1
    else:
        elbow_index = np.argmax(np.mean(accuracy, axis=1))
    scores_plot(accuracy, accuracy_std, elbow_index, cluster_range, ylabel, title, fig_filename, save_path, std_weight)


def metric_SSE(data, seed, train_centers, train_targets, val_centers, val_targets,
               save_path, dataset_name, cluster_range, runs, n_sample, number_k, number):
    # train
    scores = np.zeros((number_k, 1))        #
    for k_index, k in enumerate(cluster_range):
        centers = train_centers[k_index]
        targets = train_targets[k_index, :]
        scores[k_index] = SSE_compute(data, targets, centers)

    # val
    scores_val = np.array([])
    for r in range(runs):
        random.seed(seed + r)
        sample_index = random.sample(range(number), n_sample)
        sample_data = data[sample_index, :]
        scores_ = np.zeros(number_k)
        for k_index, k in enumerate(cluster_range):
            centers_ = val_centers[r][k_index]
            targets_ = val_targets[r][k_index, sample_index]
            scores_[k_index] = SSE_compute(sample_data, targets_, centers_)
        scores_ = np.expand_dims(scores_, axis=1)
        if r == 0:
            scores_val = scores_
        else:
            scores_val = np.concatenate((scores_val, scores_), axis=1)

    diff = np.mean((scores_val[:-1, :] - scores_val[1:, :]), axis=1)
    mean_diff = (np.mean(scores_val[0, :] - scores_val[-1, :])) / diff.shape[0]
    elbow_index = np.where(diff > 0.7 * mean_diff)[0][-1]
    elbow_k = cluster_range[elbow_index]
    print('number of cluster is {}'.format(elbow_k))

    ylabel = 'Sum of squared errors'
    fig_filename = 'SSE.jpg'
    score_filename = 'SSE.csv'
    scores_val_std = np.std(scores_val, axis=1)
    std_weight = 10
    title = dataset_name
    scores_plot(scores_val, scores_val_std, elbow_index,
                cluster_range, ylabel, title, fig_filename, save_path, std_weight)
    scores_result_save(scores, scores_val, cluster_range, score_filename, save_path)


def compute_metrics(data, subjects, save_path, cluster_range, runs, p, seed, dataset_name='SCZ'):
    number_k = np.shape(cluster_range)[0]
    number = data.shape[0]
    feature = data.shape[1]
    n_sample = round(number * p)

    train_path = save_path + os.sep + "train.pkl"
    val_path = save_path + os.sep + "val.pkl"
    with open(train_path, "rb") as f:
        train = pickle.load(f)
    with open(val_path, "rb") as f:
        val = pickle.load(f)

    train_targets = train['whole_targets']
    train_centers = train['whole_centers']
    val_targets = np.array([val[r][0] for r in range(runs)])  # (runs, number_k, number)
    val_centers = ([val[r][1] for r in range(runs)])  # (runs, ((2, feature), (3, feature), (4, feature), ...))

    metric_Stability(train_centers, train_targets, val_centers, val_targets,
                     subjects, save_path, dataset_name, cluster_range, runs, n_sample, number_k)

    metric_SSE(data, seed, train_centers, train_targets, val_centers, val_targets,
               save_path, dataset_name, cluster_range, runs, n_sample, number_k, number)
