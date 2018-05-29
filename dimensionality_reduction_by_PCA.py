#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author  : mystic
# @date    : 2018/5/27 16:35

from numpy import mat, mean, cov, linalg, argsort, shape

"""
三种降维技术:
    PCA: Principal Component Analysis(主成分分析)
    FA: Factor Analysis(因子分析)
    ICA: Independent Component Analysis(独立成分分析)
    PCA使用最广泛
    PCA:
        优点: 降低数据的复杂性, 识别最重要的多个特征
        缺点: 不一定需要, 且可能损失有用信息
        适用数据类型: 数值型数据
"""

"""
PCA:
将数据转换成前N个主成分的伪代码大致如下:
    去除平均值
    计算协方差矩阵
    计算协方差矩阵的特征值和特征向量
    将特征值降序排序
    保留前N个特征向量
    将数据转换到上述N个特征向量构建的新空间中
"""


def load_data_set(filename, delimiter='\t'):
    with open(filename) as fr:
        str_list = [line.strip().split(delimiter) for line in fr.readlines()]
        data_list = [list(map(float, line)) for line in str_list]
        return mat(data_list)


def pca(data_mat, top_n_features=9999999):
    """
        pca dimensionality reduction
    :param data_mat:
    :param top_n_features: default return all
    :return: need to return top_n_features eigen val
    """
    mean_val_mat = mean(data_mat, axis=0)
    mean_removed = data_mat - mean_val_mat
    cov_arr = cov(mean_removed, rowvar=False)
    eigen_val_arr, eigen_vector_mat = linalg.eig(mat(cov_arr))
    # ascending order, return idx,which doesn't change origin array
    eigen_val_idx_arr = argsort(eigen_val_arr)
    # Get the top N largest eigen vectors
    eigen_val_idx_arr = eigen_val_idx_arr[:-(top_n_features + 1):-1]
    red_eigen_vector_mat = eigen_vector_mat[:, eigen_val_idx_arr]
    low_dimension_data_mat = mean_removed * red_eigen_vector_mat
    # if returned all eigen vectors, reconstructed mat should be the same as input data_mat
    reconstruct_mat = (low_dimension_data_mat * red_eigen_vector_mat.T) + mean_val_mat
    return low_dimension_data_mat, reconstruct_mat


def plt_fig(data_mat, reconstruct_mat):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data_mat[:, 0].flatten().A[0], data_mat[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(reconstruct_mat[:, 0].flatten().A[0], reconstruct_mat[:, 1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()


if __name__ == '__main__':
    arr_ = [1, 4, 7671, 123, 87678, 2, 54, 87]
    sort_idx_ = argsort(arr_)
    print(arr_)
    print(sort_idx_)
    print(sort_idx_[: -4: -1])
    """
        sort_idx[start: end: n]
        The positive and negative of n 
            indicate whether it needs reverse
        the value of n, which means steps
    """
    data_mat_ = load_data_set('resource/testSet_pca.txt')
    low_dimension_mat_, reconstruct_mat_ = pca(data_mat_, 1)
    plt_fig(data_mat_, reconstruct_mat_)
    print(shape(data_mat_))
    print('=============================')
    print(shape(low_dimension_mat_))
    print(shape(reconstruct_mat_))
    print(data_mat_)
    print(reconstruct_mat_)
