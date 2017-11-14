#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by PyCharm
# @author  : mystic
# @date    : 2017/11/13 20:39
"""
    k-NearestNeighbor
    核心思想:
        如果一个样本在特征空间中的k个最相邻的样本中的大多数属于某一个类别，
        则该样本也属于这个类别，并具有这个类别上样本的特性
    优点:
        精度高、对异常值不敏感、无数据输入假定
    缺点：
        计算复杂度高、空间复杂度高
    适用数据范围：
        数值型和标称型
"""

from numpy import *
import operator


def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 1], [0, 0.1]])
    labels = ['A', 'C', 'B', 'D']
    return group, labels


"""
    in_x        用于分类的输入向量
    data_set    输入的训练样本集
    labels      标签向量
    k           用于选择最近邻居的数目
"""


def classify(in_x, data_set, labels, k):
    # 获取data_set的第一维长度
    data_set_size = data_set.shape[0]
    # 分别计算输入向量与data_set集合中各点的向量差,并存入数组中
    diff_arr = tile(in_x, (data_set_size, 1)) - data_set
    # 平方
    sq_diff_arr = diff_arr ** 2
    # 求平方和
    sq_distinces = sq_diff_arr.sum(axis=1)
    # 开根,得各点与输入向量的距离值集合
    distinces = sq_distinces ** 0.5
    # 排序,升序(返回结果为索引,如[17,23,1,0],排序后返回[3,2,0,1])
    sorted_dist_indices = distinces.argsort()
    # print('最近的点:%s' % labels[sorted_dist_indices[0]])
    # 存储最近的k个点
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_dist_indices[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    # print(class_count)
    # 根据字典class_count的value进行降序排列
    # 在最近点案例中,value都是1,下面的排序等于没做
    sorted_class_count = sorted(class_count.items(),
                                key=operator.itemgetter(1), reverse=True)
    # print(sorted_class_count)
    return sorted_class_count[0][0]


def file2matrix(filename):
    # 获取文件行数
    fr = open(filename)
    array_lines = fr.readlines()
    amount = len(array_lines)
    #
    return_matrix = zeros((amount, 3))
    class_label_vector = []
    index = 0
    for line in array_lines:
        # 截取掉回车符
        line = line.strip()
        list_from_line = line.split('\t')
        return_matrix[index, :] = list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_matrix, class_label_vector


"""
# 会输出(2,4,2)
shape([
    [
        [1, 1],
        [1, 2],
        [1, 3],
        [1, 4]
    ],
    [
        [1, 1],
        [1, 2],
        [1, 3],
        [1, 4]
    ]
])
"""



