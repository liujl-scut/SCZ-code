import os
import csv
import math
import time
import pickle
import random
import numpy as np
import matlab.engine
from scipy import stats
from scipy import special
from joblib import Parallel, delayed
from sub_result_visualize import result_visualize

eng = matlab.engine.start_matlab()
sKmeans_path = os.path.dirname(__file__)


def is_outlier(data):
    """
    数据异常值检测，具体公式参考 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9196089/
    """
    madfactor = -1 / (math.sqrt(2) * special.erfcinv(3 / 2))  # 1 / norm.ppf(3/4)
    MAD = madfactor * np.median(abs(data - np.median(data)))
    TF = (abs(data - np.median(data)) > 3 * MAD)
    return TF


def check(cdata, sub_idx, do=1):
    """数据异常值筛选"""
    if do:
        TF = is_outlier(np.sum(cdata, axis=1))  # 数据异常值检测与筛选
        inID = np.squeeze(np.array(np.where(TF == 0)))
        cdata = cdata[inID, :]
        cdata = stats.zscore(cdata, 1, ddof=1)
        sub_idx = [sub_idx[i] for i in inID]
        return cdata, sub_idx
    else:
        return cdata, sub_idx


def sparse_kmeans(data, k, s=1):
    """
    调用matlab的SparseKmeansClustering.m文件，进行稀疏Kmeans聚类算法
    """
    start_time = time.time()
    eng.cd(sKmeans_path)
    mat_data = matlab.double(data.tolist())
    out = eng.SparseKmeansClustering(mat_data, k, s, nargout=2)
    w = np.array(out[1])
    targets = np.squeeze(np.array(out[0]) - 1)
    end_time = time.time()
    t = round(end_time - start_time)
    return targets, w, t


def write_csv(subjects, targets, filename, path):
    os.chdir(path)
    headline = ['subject', 'sub_index', 'cluster']
    f = open(filename, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(headline)
    for csv_index in np.arange(len(targets)):
        csv_line = [subjects[csv_index], csv_index, targets[csv_index]]
        csv_writer.writerow(csv_line)
    f.close()


def save_result(save_path, k, split, data, targets, subjects, t):
    cluster_path = save_path + os.sep + str(k)
    if not os.path.exists(cluster_path):
        os.makedirs(cluster_path)
    write_csv(subjects, targets, split + '.csv', cluster_path)
    result_visualize(data, targets, split + '_PCA_2.jpg', cluster_path, n_components=2, metric='PCA')
    result_visualize(data, targets, split + '_PCA_3.jpg', cluster_path, n_components=3, metric='PCA')
    result_visualize(data, targets, split + '_TSNE_2.jpg', cluster_path, n_components=2, metric='TSNE')
    result_visualize(data, targets, split + '_TSNE_3.jpg', cluster_path, n_components=3, metric='TSNE')
    file_time = open(str(t) + '.txt', 'w')
    file_time.write(str(t))
    file_time.close()


def run_val(data, cluster_range, n_sample, number, number_k, feature, seed, r):
    random.seed(seed + r)
    sample_index = random.sample(range(number), n_sample)
    sample_data = data[sample_index, :]

    whole_centers = []
    whole_targets = np.full((number_k, number), np.nan)
    for k_index, k in enumerate(cluster_range):
        targets, w, t = sparse_kmeans(sample_data, k)

        whole_targets[k_index, sample_index] = targets
        centers = np.zeros((k, feature))
        for kk in range(k):
            centers[kk, :] = np.mean(sample_data[targets == kk, :], axis=0)
        whole_centers.append(centers)

    return whole_targets, whole_centers


def cluster(data, idx, save_path, class_range, seed, split, validation, runs, ratio):
    random.seed(seed)
    number_k = np.shape(class_range)[0]
    number = data.shape[0]
    feature = data.shape[1]
    n_sample = round(number * ratio)

    # trainning process
    whole_centers = []
    whole_targets = np.zeros((number_k, number))
    for k_index, k in enumerate(class_range):
        targets, w, t = sparse_kmeans(data, k)

        whole_targets[k_index, :] = targets
        centers = np.zeros((k, feature))
        for kk in range(k):
            centers[kk, :] = np.mean(data[targets == kk, :], axis=0)
        whole_centers.append(centers)

        # 二维、三维特征可视化
        save_result(save_path, k, split, data, targets, idx, t)

    # 保存聚类的数据分类以及聚类中心
    train = {"whole_targets": whole_targets, "whole_centers": whole_centers}
    train_path = save_path + os.sep + "train.pkl"
    with open(train_path, "wb") as f:
        pickle.dump(train, f)

    # cross validation
    if validation:
        # val = Parallel(n_jobs=-1)(delayed(run_val)(data, class_range, n_sample, number, number_k, feature, seed, r) for r in range(runs))
        val = [run_val(data, class_range, n_sample, number, number_k, feature, seed, r) for r in range(runs)]

        # 保存聚类的数据分类以及聚类中心
        val_path = save_path + os.sep + "val.pkl"
        with open(val_path, "wb") as f:
            pickle.dump(val, f)
