#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author  : mystic
# @date    : 2018/5/27 16:35

from numpy import mat, mean, cov, linalg

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
    mean_vals = mean(data_mat, axis=0)
    mean_removed = data_mat - mean_vals
    cov_arr = cov(mean_removed, rowvar=False)
    eigen_vals, eigen_vectors = linalg.eig(mat(cov_arr))
    pass
