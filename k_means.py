#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author  : mystic
# @date    : 2018/5/6 17:11

"""
        K-均值聚类的一般流程
1. 收集数据: 使用任意方法
2. 准备数据: 需要数值型数据来计算距离, 也可以将标称型数据映射为二值型数据, 再用于计算
3. 分析数据: 使用任意方法
4. 训练算法: 不适用于无监督学习, 即无监督学习没有训练过程
5. 测试算法: 应用聚类算法, 观察结果
            可以使用量化的误差指标,如误差平方和来评价算法的结果
6. 使用算法: 可以用于所希望的任何应用
            通常情况下, 簇质心可以代表整个簇的数据来作出决策
"""
from numpy import shape, zeros, mat, random, inf, nonzero, mean
from numpy.linalg import linalg


def load_data_set(filename):  # general function to parse tab -delimited floats
    data_list = []  # assume last column is target value
    with open(filename) as fr:
        for line in fr.readlines():
            cur_line = line.strip().split('\t')
            # TODO the return type has changed in python3
            # before change: map(float, cur_line)
            # after change: list(map(float, cur_line))
            flt_line = list(map(float, cur_line))  # map all elements to float()
            data_list.append(flt_line)
        return data_list


def euclidean_distance(vector_a, vector_b):
    # return sqrt(sum(power(vector_a - vector_b, 2)))
    # The L2 norm is the Euclidean Distance
    return linalg.norm(vector_a - vector_b)


def rand_centroid(data_set, k):
    n = shape(data_set)[1]
    centroid_mat = mat(zeros((k, n)))
    # create random cluster centers, within bounds of each dimension
    for j in range(n):
        min_j = min(data_set[:, j])
        range_j = float(max(data_set[:, j]) - min_j)
        centroid_mat[:, j] = min_j + range_j * random.rand(k, 1)
    return centroid_mat


def k_means(data_set, k, distance_measure=euclidean_distance, create_centroid=rand_centroid):
    m = shape(data_set)[0]
    # create mat to assign data points
    # to a centroid, also holds SE of each point
    cluster_assignment = mat(zeros((m, 2)))
    centroid_mat = create_centroid(data_set, k)
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        # for each data point assign it to the closest centroid
        for i in range(m):
            min_dist = inf
            min_idx = -1
            for j in range(k):
                dist_j2i = distance_measure(centroid_mat[j, :], data_set[i, :])
                if dist_j2i < min_dist:
                    min_dist = dist_j2i
                    min_idx = j
            if cluster_assignment[i, 0] != min_idx:
                cluster_changed = True
            cluster_assignment[i, :] = min_idx, min_dist ** 2
        print(centroid_mat)
        # recalculate centroids
        for cent in range(k):
            # get all the point in this cluster
            points_in_cluster = data_set[nonzero(cluster_assignment[:, 0].A == cent)[0]]
            # assign centroid to mean
            centroid_mat[cent, :] = mean(points_in_cluster, axis=0)
    return centroid_mat, cluster_assignment


def binary_k_means(data_set, k, distance_measure=euclidean_distance):
    m = shape(data_set)[0]
    # create mat to assign data points
    # to a centroid, also holds SE of each point
    cluster_assignment = mat(zeros((m, 2)))
    centroid = list(mean(data_set, axis=0))[0]
    # create a list with one centroid
    centroid_list = list(centroid)
    # calc initial Error
    for j in range(m):
        cluster_assignment[j, 1] = distance_measure(mat(centroid_list), data_set[j, :]) ** 2
    while len(centroid_list) < k:
        lowest_sse = inf
        best_centroid2split = None
        best_new_centroid = None
        best_cluster_assignment = None
        for i in range(len(centroid_list)):
            # get the data points currently in cluster i
            points_in_current_cluster = data_set[nonzero(cluster_assignment[:, 0].A == i)[0], :]
            centroid_mat, split_cluster_assignment = k_means(points_in_current_cluster, 2, distance_measure)
            # compare the SSE to the current minimum
            sse_split = sum(split_cluster_assignment[:, 1])
            sse_not_split = sum(cluster_assignment[nonzero(cluster_assignment[:, 0].A != i)[0], 1])
            print('SSE_Split, and not split: ', sse_split, sse_not_split)
            if sse_split + sse_not_split < lowest_sse:
                best_centroid2split = i
                best_new_centroid = centroid_mat
                best_cluster_assignment = split_cluster_assignment.copy()
                lowest_sse = sse_split + sse_not_split
        # change 1 to 3,4, or whatever
        best_cluster_assignment[nonzero(best_cluster_assignment[:, 0].A == 1)[0], 0] = len(centroid_list)
        best_cluster_assignment[nonzero(best_cluster_assignment[:, 0].A == 0)[0], 0] = best_centroid2split
        print('The best_centroid2split is: ', best_centroid2split)
        print('The length of best_cluster_assignment is: ', len(best_cluster_assignment))
        # replace a centroid with two best centroids
        centroid_list[best_centroid2split] = best_new_centroid[0, :]
        centroid_list.append(best_new_centroid[1, :])
        # reassign new clusters, and SSE
        cluster_assignment[nonzero(cluster_assignment[:, 0].A == best_centroid2split)[0], :] = best_cluster_assignment
    return mat(centroid_list), cluster_assignment


if __name__ == '__main__':
    data_mat_ = mat(load_data_set('resource/testSet.txt'))
    my_centroid_mat_, cluster_assignment_ = k_means(data_mat_, 4)
    print(my_centroid_mat_)
    print('=======================================')
    print(cluster_assignment_)
