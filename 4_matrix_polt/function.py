import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from PIL import Image


# 进行图片的复制拼接
def concat_images(image_names, name, path, COL, ROW, UNIT_HEIGHT_SIZE, UNIT_WIDTH_SIZE, SAVE_QUALITY):
    image_files = []
    for index in range(COL * ROW):
        image_files.append(Image.open(path + image_names[index]))  # 读取所有用于拼接的图片
    target = Image.new('RGB', (UNIT_WIDTH_SIZE * COL, UNIT_HEIGHT_SIZE * ROW))  # 创建成品图的画布
    # 第一个参数RGB表示创建RGB彩色图，第二个参数传入元组指定图片大小，第三个参数可指定颜色，默认为黑色
    for row in range(ROW):
        for col in range(COL):
            # 对图片进行逐行拼接
            # paste方法第一个参数指定需要拼接的图片，第二个参数为二元元组（指定复制位置的左上角坐标）
            # 或四元元组（指定复制位置的左上角和右下角坐标）
            target.paste(image_files[COL * row + col], (0 + UNIT_WIDTH_SIZE * col, 0 + UNIT_HEIGHT_SIZE * row))
    target.save(path + name + '.jpg', quality=SAVE_QUALITY)  # 成品图保存


def plot_matrix_31x31(mat_EC, mat_EO, minV, maxV, path, band):
    # 设置画布与坐标轴
    f, _ = plt.subplots(2, 2, figsize=(20, 20))
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)

    # 归一化色2条范围
    norm = matplotlib.colors.Normalize(vmin=minV, vmax=maxV)

    # 设置蒙版
    mask = np.zeros_like(mat_EC[0], dtype=np.bool_)  # 构造与corr同维数矩阵为bool型
    mask[np.diag_indices_from(mask)] = True  # 设置主对角线元素为True

    # annot=False不显示热图块上的数值，cbar:colorbar cmap 热图颜色 norm 设置统一的颜色量度
    camp = "RdBu_r"  # RdBu_r,afmhot,gist_heat,hot
    sns.heatmap(mat_EC[0], square=True, cmap=camp, ax=ax1, cbar=False, norm=norm, mask=mask)  # linewidths=0.3
    sns.heatmap(mat_EC[1], square=True, cmap=camp, ax=ax2, cbar=False, norm=norm, mask=mask)
    sns.heatmap(mat_EO[0], square=True, cmap=camp, ax=ax3, cbar=False, norm=norm, mask=mask)
    sns.heatmap(mat_EO[1], square=True, cmap=camp, ax=ax4, cbar=False, norm=norm, mask=mask)

    # 添加水平线和竖直网格线
    line = [7, 11, 13, 14, 15, 17, 21, 28, 29, 30]
    ax = [ax1, ax2, ax3, ax4]
    for i in line:
        # 指定每条线的位置、颜色和线宽
        for x in ax:
            x.axhline(i, color='k', ls='--')
            x.axvline(i, color='k', ls='--')
    for i in [0, 31]:
        for x in ax:
            x.axhline(i, color='k', linewidth=8)
            x.axvline(i, color='k', linewidth=8)

    # 设置蒙版（主对角线元素）为黑色
    for x in ax:
        x.set_facecolor('xkcd:black')

    # 设置ax1标签
    ticks = [3.5, 9, 12, 13.5, 14.5, 16, 19, 24.5, 28.5, 29.5, 30.5]
    ticklabel = ['L frontal lobe', 'L parietal lobe', 'L temporal lobe', 'L visual cortex',
                 'R visual cortex', 'R temporal lobe', 'R parietal lobe', 'R frontal lobe',
                 'MPFC', 'PCC', 'DACC']
    ax1.set(xticks=ticks)
    ax1.set(yticks=ticks)
    ax1.set_xticklabels(ticklabel, rotation=90, fontsize=10)
    ax1.set_yticklabels(ticklabel, rotation=0, fontsize=10)

    # X x轴设置不可见
    ax2.xaxis.set_visible(False)
    ax3.xaxis.set_visible(False)
    ax4.xaxis.set_visible(False)

    # Y y轴设置不可见
    ax2.yaxis.set_visible(False)
    ax3.yaxis.set_visible(False)
    ax4.yaxis.set_visible(False)

    # 设置title
    sns.set(font_scale=1.5)
    f.suptitle(band, fontsize=30)
    ax1.set_title('EC cluster 1')
    ax2.set_title('EC cluster 2')
    ax3.set_title('EO cluster 1')
    ax4.set_title('EO cluster 2')

    # 定义一个位置
    position = f.add_axes([0.92, 0.11, 0.03, 0.77])  # [位置：x,y 大小：长，宽]
    f.colorbar(
        # 对应颜色与色值映射
        matplotlib.cm.ScalarMappable(cmap=camp, norm=norm),
        # 注意坐标轴传递的格式
        ax=np.array([[ax1, ax2]]).ravel().tolist(),
        # 设置位置
        cax=position, )

    plt.savefig(path + band + '.jpg')
    plt.cla()
    plt.clf()
    plt.close()


def plot_ttest(mat, path, name):
    # 设置画布与坐标轴
    f, _ = plt.subplots(2, 3, figsize=(30, 20))
    ax1 = plt.subplot(231)
    ax2 = plt.subplot(232)
    ax3 = plt.subplot(233)
    ax4 = plt.subplot(234)
    ax5 = plt.subplot(235)
    ax6 = plt.subplot(236)

    # 归一化色2条范围
    norm = matplotlib.colors.Normalize(vmin=0)

    # 产生一个上三角和主对角线是Ture，下三角是False的31*31的numpy矩阵
    mask = np.zeros((31, 31), dtype=bool)
    indices = np.triu_indices(31, k=0)
    mask[indices] = True
    np.fill_diagonal(mask, True)

    # annot=False不显示热图块上的数值，cbar:colorbar cmap 热图颜色 norm 设置统一的颜色量度
    camp = "hot_r"  # RdBu_r,afmhot,gist_heat,hot
    sns.heatmap(mat[0], square=True, cmap=camp, ax=ax1, cbar=False, norm=norm, mask=mask)  # linewidths=0.3
    sns.heatmap(mat[1], square=True, cmap=camp, ax=ax2, cbar=False, norm=norm, mask=mask)
    sns.heatmap(mat[2], square=True, cmap=camp, ax=ax3, cbar=False, norm=norm, mask=mask)
    sns.heatmap(mat[3], square=True, cmap=camp, ax=ax4, cbar=False, norm=norm, mask=mask)
    sns.heatmap(mat[4], square=True, cmap=camp, ax=ax5, cbar=False, norm=norm, mask=mask)
    sns.heatmap(mat[5], square=True, cmap=camp, ax=ax6, cbar=False, norm=norm, mask=mask)

    # 添加水平线和竖直网格线
    ax = [ax1, ax2, ax3, ax4, ax5, ax6]
    for x in ax:
        for i in range(31):
            x.axhline(i, xmin=0, xmax=(i + 1) / 31, color='k', ls='--', linewidth=0.3)
            x.axvline(31 - i, ymin=0, ymax=(i + 1) / 31, color='k', ls='--', linewidth=0.3)
        x.axhline(31, color='k', linewidth=1)
        x.axvline(0, color='k', linewidth=1)

    line = [0, 7, 11, 13, 14, 15, 17, 21, 28, 29, 30, 31]
    for x in ax:
        for i in line:
            x.axhline(i, xmin=0, xmax=i / 31, color='k', linewidth=0.5)
            x.axvline(i, ymin=0, ymax=(31 - i) / 31, color='k', linewidth=0.5)

    # 设置蒙版背景颜色
    for x in ax:
        x.set_facecolor('xkcd:white')

    # 设置ax1标签
    ticks = [3.5, 9, 12, 13.5, 14.5, 16, 19, 24.5, 28.5, 29.5, 30.5]
    ticklabel = ['L frontal lobe', 'L parietal lobe', 'L temporal lobe', 'L visual cortex',
                 'R visual cortex', 'R temporal lobe', 'R parietal lobe', 'R frontal lobe',
                 'MPFC', 'PCC', 'DACC']
    ax1.set(xticks=ticks)
    ax1.set(yticks=ticks)
    ax1.set_xticklabels(ticklabel, rotation=90, fontsize=10)
    ax1.set_yticklabels(ticklabel, rotation=0, fontsize=10)

    # X x轴设置不可见
    ax2.xaxis.set_visible(False)
    ax3.xaxis.set_visible(False)
    ax4.xaxis.set_visible(False)
    ax5.xaxis.set_visible(False)
    ax6.xaxis.set_visible(False)

    # Y y轴设置不可见
    ax2.yaxis.set_visible(False)
    ax3.yaxis.set_visible(False)
    ax4.yaxis.set_visible(False)
    ax5.yaxis.set_visible(False)
    ax6.yaxis.set_visible(False)

    # 设置title
    sns.set(font_scale=1.5)
    f.suptitle(name, fontsize=30)
    ax1.set_title('all')
    ax2.set_title('Delta')
    ax3.set_title('Theta')
    ax4.set_title('Alpha')
    ax5.set_title('Beta')
    ax6.set_title('Gamma')

    # 定义一个位置
    position = f.add_axes([0.92, 0.11, 0.03, 0.77])  # [位置：x,y 大小：长，宽]
    f.colorbar(
        # 对应颜色与色值映射
        matplotlib.cm.ScalarMappable(cmap=camp, norm=norm),
        # 注意坐标轴传递的格式
        ax=np.array([[ax1, ax2]]).ravel().tolist(),
        # 设置位置
        cax=position, )

    plt.savefig(path + name + '.jpg')
    plt.cla()
    plt.clf()
    plt.close()


def convert_feature465_matrix31x31(data, df, band, z_score=False):
    """
    使用聚类结果（df）的id号作为索引，使用该索引对数据data进行筛选，对筛选后的数据进行转化，
    转化方式为将465的数据特征转化为31*31的矩阵，
    矩阵主对角线元素为0，矩阵是对称矩阵。
    分别转化出class 1和class 0的特征矩阵，然后分别求出class 1和class 0的特征矩阵的均值
    (input)
    data: 输入维度为465的特征数据（EC/EO，pec/wpli/icoh）
    df:   聚类结果
    band: 选择的频段
    (output)
    result_mat：存放class0以及class1的功能连接矩阵（均值）
    max_value:  存放class0/class1功能连接矩阵的最大值
    min_value:  存放class0/class1功能连接矩阵的最小值
    """
    fea = [0, 0]  # 存放class0以及class1的特征
    result_mat = [0, 0]  # 存放class0以及class1的功能连接矩阵（均值）
    max_value = 0  # 存放class0\class1功能连接矩阵的最大值
    min_value = 100  # 存放class0\class1功能连接矩阵的最小值
    for class_ in range(2):  # 遍历class0和class1
        sub_number = data['ROI'][0, 0]['sub_number']  # 存放data中所有被试者的id号
        data_band = data['ROI'][0, 0]['overall'][0, 0][band]  # 取频段为band的ROI data
        sub_cluster = df.loc[df['cluster'] == class_]['subject']  # 存放class0\class1中被试者的id号（'cluster'列）
        if z_score:
            data_band = stats.zscore(data_band, 1, ddof=1)

        index = []
        for idx in sub_cluster:  # 将class0\class1中被试者的id号，转化为data中的index
            index.append(np.argwhere(sub_number == idx)[0, 0])
        data_cluster = data_band[index, :]  # (number, 465) 通过class0\class1中被试者的id号，来取出对应的ROI data
        fea[class_] = data_cluster

        c_mat = np.zeros([31, 31])  # 存放其中一类所有功能连接矩阵的总和
        num = data_cluster.shape[0]  # class0\class1中被试者的数量
        for i in range(num):
            ROIData = data_cluster[i, :]  # shape: (465,)  遍历所有被试者的ROI data
            mat = np.zeros([31, 31])  # 存放一个被试者的功能连接矩阵（31*31）
            n = 0
            for y in range(0, 30):  # 将465个特征按照坐标位置，转化成31*31矩阵
                for x in range(y + 1, 31):
                    mat[x][y] = ROIData[n]  # 构造功能链接矩阵下三角
                    mat[y][x] = ROIData[n]  # 功能连接矩阵是对称的，构造上三角
                    n = n + 1
            c_mat = c_mat + mat
        c_mean_mat = c_mat / num  # 求class0\class1功能连接矩阵的均值
        result_mat[class_] = c_mean_mat
        if c_mean_mat.max() > max_value:
            max_value = c_mean_mat.max()
        if c_mean_mat.min() < min_value:
            min_value = c_mean_mat.min()
    return result_mat, max_value, min_value, fea


def ttest(feature465):
    class0 = feature465[0]
    class1 = feature465[1]
    tvalue = np.zeros([465])
    diffvalue = np.zeros([465])
    for i in range(465):
        _, p = stats.levene(class0[:, i], class1[:, i])  # 进行levene检验
        if p > 0.05:
            t, pvalue = stats.ttest_ind(class0[:, i], class1[:, i], equal_var=True)
        else:
            t, pvalue = stats.ttest_ind(class0[:, i], class1[:, i], equal_var=False)
        if pvalue < 0.05:
            tvalue[i] = np.abs(t)
            diffvalue[i] = np.abs(class0[:, i].mean() - class1[:, i].mean())

    tmat = np.zeros([31, 31])
    diffmat = np.zeros([31, 31])
    n = 0
    for y in range(0, 30):  # 将465个特征按照坐标位置，转化成31*31矩阵
        for x in range(y + 1, 31):
            tmat[x][y] = tvalue[n]  # 构造矩阵下三角
            diffmat[x][y] = diffvalue[n]  # 构造矩阵下三角
            n = n + 1
    return tmat, diffmat
