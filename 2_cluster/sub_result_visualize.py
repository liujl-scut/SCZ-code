import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import levene
from statsmodels.formula.api import ols
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold._t_sne import TSNE
from sklearn.decomposition._pca import PCA
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)


def result_visualize(data, targets, filename, path, n_components=2, metric='PCA'):
    if metric == 'PCA':
        model = PCA(n_components=n_components)  # 降到2维
    elif metric == 'TSNE':
        model = TSNE(n_components=n_components, random_state=0)

    data_new = model.fit_transform(data)
    colors = ['r', 'g', 'b', 'y', 'c', 'm']

    if n_components == 2:
        os.chdir(path)
        plt.figure(figsize=(6, 6), dpi=200)
        X = data_new[:, 0]
        y = data_new[:, 1]
        x_min, x_max = X.min() - (X.max() - X.min()) * 0.1, X.max() + (X.max() - X.min()) * 0.1
        y_min, y_max = y.min() - (y.max() - y.min()) * 0.1, y.max() + (y.max() - y.min()) * 0.1
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title('Clusters', fontsize=14)
        # 绘制散点图
        g = [None] * 6
        cluster_labels = [None] * 6
        for target_index in set(targets):
            target_index = int(target_index)
            index = np.where(targets == target_index)[0]
            g[target_index] = plt.scatter(X[index], y[index], s=50, c=colors[target_index], edgecolors='k',
                                          linewidths=0.8)
            cluster_labels[target_index] = 'Cluster ' + str(target_index + 1)

        plt.legend(handles=g[:target_index + 1], labels=cluster_labels[:target_index + 1])

        plt.savefig(filename)
        plt.savefig(filename[:-4] + '.svg')

    elif n_components == 3:
        X = data_new[:, 0]
        y = data_new[:, 1]
        z = data_new[:, 2]
        os.chdir(path)
        fig = plt.figure(figsize=(6, 6), dpi=200)
        ax = Axes3D(fig)
        ax.set_title('Clusters', fontsize=14)
        # 绘制散点图
        for target_index in set(targets):
            target_index = int(target_index)
            index = np.where(targets == target_index)[0]
            ax.scatter(X[index], y[index], z[index], s=20, c=colors[target_index], edgecolors='k', linewidths=0.8)
        plt.savefig(filename)

    else:
        print('Can not visualize!')


def indices_plot(indices_data, indices, clusters, path):
    fig_path = path + '/' + 'IndicesFigures'
    data_path = path + '/' + 'Indices'
    statistic_path = path + '/' + 'Statistic'
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    if not os.path.exists(statistic_path):
        os.mkdir(statistic_path)

    for indices_index in range(len(indices)):
        y_name = indices[indices_index]
        y = [indices_data[i][:, indices_index] for i in range(len(clusters))]
        max_n = np.max([np.shape(indices_data[i])[0] for i in range(len(clusters))])
        y_array = np.full((len(clusters), max_n), np.nan)
        for i in range(len(clusters)):
            n_cluster_sample = np.shape(indices_data[i])[0]
            y_array[i, :n_cluster_sample] = indices_data[i][:, indices_index]

        y_mean = np.array([np.mean(indices_data[i][:, indices_index]) for i in range(len(clusters))])
        y_sem = np.array([np.std(indices_data[i][:, indices_index]) / np.sqrt(np.shape(indices_data[i])[0]) for i in
                          range(len(clusters))])
        ymax = np.max(y_mean + y_sem)
        ymin = np.min(y_mean - y_sem)
        x = np.arange(len(clusters)) + 1
        k = len(clusters)

        if len(y) > 1:
            x_start = []
            x_end = []
            y_start = []
            y_end = []
            pvalues = []
            y_interval = 0.1

            groups = np.tile(np.arange(y_array.shape[0]), y_array.shape[1])
            y_flat = y_array.T.flatten()
            groups_flat = groups[~np.isnan(y_flat)]
            y_flat = y_flat[~np.isnan(y_flat)]

            stat, pp = levene(*y)

            if k == 2:

                # stats.wilcoxon(y_flat[groups_flat == 0], y_flat[groups_flat == 1])
                if pp >= 0.05:
                    states, pp_k = stats.ttest_ind(y_flat[groups_flat == 0], y_flat[groups_flat == 1])
                else:
                    states, pp_k = stats.ttest_ind(y_flat[groups_flat == 0], y_flat[groups_flat == 1], equal_var=False)
                Results = pd.DataFrame([states, pp])
                os.chdir(statistic_path)
                filename = 'No_' + str(k) + '_' + y_name + '.xlsx'
                writer = pd.ExcelWriter(filename)
                Results.to_excel(excel_writer=writer, sheet_name='anovaResults')
                writer.save()
                writer.close()

                if pp_k < 0.05:
                    x_start.append(x[0] + 0.1)
                    x_end.append(x[1] - 0.1)
                    y_start.append(ymax + y_interval * ymax)
                    pvalues_sort = [pp]

            else:

                os.chdir(statistic_path)
                filename = 'No_' + str(k) + '_' + y_name + '.xlsx'
                writer = pd.ExcelWriter(filename)

                df = np.rec.array([(str(groups_flat[i]), y_flat[i]) for i in range(groups_flat.shape[0])],
                                  dtype=[('clusters', '|S1'), ('Indices', '<i4')])

                if pp >= 0.05:
                    model = ols('Indices ~ clusters', df).fit()
                    anovaResults = anova_lm(model, typ=2)
                    anovaResults.to_excel(excel_writer=writer, sheet_name='anovaResults')
                    pp_k = anovaResults['PR(>F)'][0]
                else:
                    states, pp_k = stats.kruskal(*y)
                    kruskalResults = pd.DataFrame([states, pp_k])
                    kruskalResults.to_excel(excel_writer=writer, sheet_name='kruskalResults')

                if pp_k < 0.05:
                    multiComp = MultiComparison(df['Indices'], df['clusters'])
                    # rtp = multiComp.allpairtest(stats.ttest_ind, method='bonferroni')
                    rtp = multiComp.allpairtest(stats.ttest_ind, method='fdr_bh')
                    print((rtp[0]))
                    pvalues = rtp[1][2]

                    # tukeyhsd_test = multiComp.tukeyhsd()
                    # pvalues = tukeyhsd_test.pvalues
                    pairindices = multiComp.pairindices
                    pair_temp = pairindices[1] - pairindices[0]
                    pair_sort_index = np.argsort(pair_temp)
                    pvalues_sort = pvalues[pair_sort_index]
                    for index in range(len(pvalues_sort)):
                        if pvalues_sort[index] < 0.05:
                            x_start.append(x[pairindices[0][pair_sort_index[index]]] + 0.1)
                            x_end.append(x[pairindices[1][pair_sort_index[index]]] - 0.1)
                            if not y_start:
                                y_start.append(ymax + y_interval * ymax)
                            else:
                                if pairindices[1][pair_sort_index[index]] - pairindices[0][pair_sort_index[index]] == 1:
                                    y_start.append(y_start[-1])
                                else:
                                    y_start.append(y_start[-1] + y_interval * ymax)

                    MultiComparisonResults = pd.DataFrame(rtp)
                    MultiComparisonResults.to_excel(excel_writer=writer, sheet_name='MultiComparisonResults')
                writer.save()
                writer.close()

            fig = plt.figure(figsize=(4, 6))
            ax = fig.add_subplot(111)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks(np.arange(1, k + 1))
            ax.spines['bottom'].set_linewidth(1.5)
            ax.spines['left'].set_linewidth(1.5)
            plt.tick_params(labelsize=12, width=2)
            plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
            color_range = np.arange(0, 1 + 1 / (x.shape[0] - 1), 1 / (x.shape[0] - 1))
            # color_range = np.log(range(1, x.shape[0] + 1)) / np.log(x.shape[0])
            color_range = [str(color_range[i]) for i in range(len(color_range))]
            ax.bar(x, y_mean, width=0.5, edgecolor='black', linewidth=1.5, color=color_range)
            plt.errorbar(x, y_mean, yerr=y_sem, linestyle='', elinewidth=1.5, ecolor='black', capsize=3)
            plt.xlim((0.5, k + 0.5))
            if not y_start:
                plt.ylim((0, ymax + y_interval * ymax))
            else:
                plt.ylim((0, y_start[-1] + y_interval * ymax))

            for p_index in range(len(x_start)):
                plt.plot([x_start[p_index], x_end[p_index]], [y_start[p_index], y_start[p_index]], color="black",
                         linewidth=1.5)
                if (pvalues_sort[p_index] <= 0.05) & (pvalues_sort[p_index] > 0.01):
                    text = r'$*$'
                elif (pvalues_sort[p_index] <= 0.01) & (pvalues_sort[p_index] > 0.001):
                    text = r'$*$$*$'
                elif (pvalues_sort[p_index] <= 0.001) & (pvalues_sort[p_index] > 0.0001):
                    text = r'$*$$*$$*$'
                else:
                    text = r'$*$$*$$*$$*$'
                plt.annotate(text,
                             xy=((x_start[p_index] + x_end[p_index]) / 2, y_start[p_index] + 0.01 * y_interval * ymax),
                             fontsize=16, color="black", ha='center')

            plt.xlabel('Cluster', fontsize=16)
            plt.ylabel(y_name, fontsize=16)
            plt.title('No. clusters = ' + str(k), fontsize=20)
            os.chdir(fig_path)
            filename = 'No_' + str(k) + '_' + y_name + '.jpg'
            plt.savefig(filename)
            # plt.show()

        filename = 'No_' + str(k) + '_' + y_name + '.csv'
        os.chdir(data_path)
        f = open(filename, 'w', encoding='utf-8', newline='')
        csv_writer = csv.writer(f)
        # y_zip = zip(*y)
        # a=np.array(y)
        # for row in y_zip:
        #     csv_writer.writerow(row)
        csv_writer.writerow(np.arange(1, k + 1))
        for csv_index in range(np.shape(y_array)[1]):
            csv_writer.writerow(y_array[:, csv_index])
        f.close()
