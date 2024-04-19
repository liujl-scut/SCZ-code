import os
import pandas as pd
from function import define_cluster_label, overall_split, sub_visual, sub_id, chi_square, diff, write_csv


SCZ = 'lishui'
path = 'D:/myTask/cluster/result/' + SCZ + os.sep
feature = ['pec', 'wpli', 'icoh']
split_half = ['overall', 'split1', 'split2']
bandnames = ['merge', 'DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA', 'all']

# 结果保存在csv中
csv_list1 = [[1, 2, 3, 4]]
csv_list2 = [[1, 2, 3, 4]]
csv_list3 = [[1, 2, 3, 4]]

# 计算同一特征下，相同频段样本聚类的折半实验重复度（overall, split1, split2）
if csv_list1:
    for f in feature:
        print(f)
        for b in bandnames:
            print(b)
            label_path = path + f + os.sep + b + os.sep + '2' + os.sep
            csv_path1 = label_path + 'overall.csv'
            csv_path2 = label_path + 'split1.csv'
            csv_path3 = label_path + 'split2.csv'
            oa = pd.read_csv(csv_path1)
            s1 = pd.read_csv(csv_path2)
            s2 = pd.read_csv(csv_path3)

            # 重新定义label，个数少的为label0，个数多的为label1
            oa = define_cluster_label(oa)
            s1 = define_cluster_label(s1)
            s2 = define_cluster_label(s2)

            # 以oa为基准，s中与o相同个数多的判断为同一类
            s1 = overall_split(oa, s1)
            s2 = overall_split(oa, s2)

            # sub Label可视化：绘制overall、split1、split2的聚类结果
            sub_visual(oa, s1, s2, label_path)

            # 统计两个类别总数
            oa_c0, oa_c1 = sub_id(oa)
            s1_c0, s1_c1 = sub_id(s1)
            s2_c0, s2_c1 = sub_id(s2)
            oa_sum = len(oa_c0) + len(oa_c1)
            s1_sum = len(s1_c0) + len(s1_c1)
            s2_sum = len(s2_c0) + len(s2_c1)
            print(len(oa_c0), len(oa_c1), oa_sum)
            print(len(s1_c0), len(s1_c1), s1_sum)
            print(len(s2_c0), len(s2_c1), s2_sum)
            csv_list1.append([f])
            csv_list1.append([b, 'c0', 'c1', 'SUM'])
            csv_list1.append(['overall', len(oa_c0), len(oa_c1), oa_sum])
            csv_list1.append(['half1', len(s1_c0), len(s1_c1), s1_sum])
            csv_list1.append(['half2', len(s2_c0), len(s2_c1), s2_sum])

            # 统计卡方检验
            oa_s1_sta, oa_s1_p = chi_square(oa_c0, oa_c1, s1_c0, s1_c1)
            oa_s2_sta, oa_s2_p = chi_square(oa_c0, oa_c1, s2_c0, s2_c1)
            s1_s2_sta, s1_s2_p = chi_square(s1_c0, s1_c1, s2_c0, s2_c1)
            print('{:.2f}, {:.2f}'.format(oa_s1_sta, oa_s1_p))
            print('{:.2f}, {:.2f}'.format(oa_s2_sta, oa_s2_p))
            print('{:.2f}, {:.2f}'.format(s1_s2_sta, s1_s2_p))
            csv_list1.append([''])
            csv_list1.append(['Chi Square', 'statistic', 'pvalue'])
            csv_list1.append(['overall-half1', '{:.2f}'.format(oa_s1_sta), '{:.2f}'.format(oa_s1_p)])
            csv_list1.append(['overall-half2', '{:.2f}'.format(oa_s2_sta), '{:.2f}'.format(oa_s2_p)])
            csv_list1.append(['half1-half2', '{:.2f}'.format(s1_s2_sta), '{:.2f}'.format(s1_s2_p)])

            # 统计差别个数
            oa_s1 = diff(oa_c0, s1_c0)
            oa_s2 = diff(oa_c0, s2_c0)
            s1_s2 = diff(s1_c0, s2_c0)
            print(oa_s1, oa_s2, s1_s2)
            print()
            csv_list1.append([''])
            csv_list1.append(['', '变化的被试个数', '同样类别被试个数占比（%）'])
            csv_list1.append(['overall-half1', oa_s1, '{:.2f}'.format((1 - oa_s1 / oa_sum) * 100)])
            csv_list1.append(['overall-half2', oa_s2, '{:.2f}'.format((1 - oa_s2 / oa_sum) * 100)])
            csv_list1.append(['half1-half2', s1_s2, '{:.2f}'.format((1 - s1_s2 / s1_sum) * 100)])
            csv_list1.append([''])
            csv_list1.append([''])
    write_csv(path + 'overlap1.csv', csv_list1)

# 计算同一特征下，不同频段的样本聚类的重复度（overall）
if csv_list2:
    for f in feature:
        csv_list2.append([f])
        print(f)
        for i, b1 in enumerate(bandnames):
            for b2 in bandnames[i + 1:]:
                csv_list2.append(['', 'class0', 'class1', 'sum'])
                oPath1 = path + f + os.sep + b1 + '/2/overall.csv'
                oPath2 = path + f + os.sep + b2 + '/2/overall.csv'
                oa1 = pd.read_csv(oPath1)
                oa2 = pd.read_csv(oPath2)
                print(b1)
                print(b2)

                # 以第一个输入oa1为基准，判断oa2的聚类，将oa1与oa2更多相同标签的情况作为oa2的聚类标签
                oa2 = overall_split(oa1, oa2)

                # 统计两个类别总数
                oa1_c0, oa1_c1 = sub_id(oa1)
                oa2_c0, oa2_c1 = sub_id(oa2)
                oa1_sum = len(oa1_c0) + len(oa1_c1)
                oa2_sum = len(oa2_c0) + len(oa2_c1)
                csv_list2.append([str(b1), len(oa1_c0), len(oa1_c1), oa1_sum])
                csv_list2.append([str(b2), len(oa2_c0), len(oa2_c1), oa2_sum])
                csv_list2.append([''])
                print(len(oa1_c0), len(oa1_c1))
                print(len(oa2_c0), len(oa2_c1))

                # 统计卡方检验
                sta, p = chi_square(oa1_c0, oa1_c1, oa2_c0, oa2_c1)
                csv_list2.append(['Chi Square-statistic', '{:.2f}'.format(sta)])
                csv_list2.append(['Chi Square-pvalue', '{:.2f}'.format(p)])
                csv_list2.append([''])
                print('{:.2f}, {:.2f}'.format(sta, p))

                # 统计差别个数
                d = diff(oa1_c0, oa2_c0)
                csv_list2.append(['类别变化的被试个数', d])
                csv_list2.append(['相同类别的被试个数占比（%）', '{:.2f}'.format((1 - d / oa1_sum) * 100)])
                csv_list2.append([])
                print(d)
                print()
    write_csv(path + 'overlap2.csv', csv_list2)

# 计算同一频段下，不同特征的样本聚类的重复度（overall）
if csv_list3:
    for b in bandnames:
        csv_list3.append([b])
        print(b)
        for i, f1 in enumerate(feature):
            for f2 in feature[i + 1:]:
                csv_list3.append(['', 'class0', 'class1', 'sum'])
                oPath1 = path + f1 + os.sep + b + '/2/overall.csv'
                oPath2 = path + f2 + os.sep + b + '/2/overall.csv'
                oa1 = pd.read_csv(oPath1)
                oa2 = pd.read_csv(oPath2)
                print(f1)
                print(f2)

                # 以第一个输入oa1为基准，判断oa2的聚类，将oa1与oa2更多相同标签的情况作为oa2的聚类标签
                oa2 = overall_split(oa1, oa2)

                # 统计两个类别总数
                oa1_c0, oa1_c1 = sub_id(oa1)
                oa2_c0, oa2_c1 = sub_id(oa2)
                oa1_sum = len(oa1_c0) + len(oa1_c1)
                oa2_sum = len(oa2_c0) + len(oa2_c1)
                csv_list3.append([f1, len(oa1_c0), len(oa1_c1), oa1_sum])
                csv_list3.append([f2, len(oa2_c0), len(oa2_c1), oa2_sum])
                csv_list3.append([''])
                print(len(oa1_c0), len(oa1_c1))
                print(len(oa2_c0), len(oa2_c1))

                # 统计卡方检验
                sta, p = chi_square(oa1_c0, oa1_c1, oa2_c0, oa2_c1)
                csv_list3.append(['Chi Square-statistic', '{:.2f}'.format(sta)])
                csv_list3.append(['Chi Square-pvalue', '{:.2f}'.format(p)])
                csv_list3.append([''])
                print('{:.2f}, {:.2f}'.format(sta, p))

                # 统计差别个数
                d = diff(oa1_c0, oa2_c0)
                csv_list3.append(['类别变化的被试个数', d])
                csv_list3.append(['相同类别的被试个数占比（%）', '{:.2f}'.format((1 - d / oa1_sum) * 100)])
                csv_list3.append([])
                print(d)
                print()
    write_csv(path + 'overlap3.csv', csv_list3)
