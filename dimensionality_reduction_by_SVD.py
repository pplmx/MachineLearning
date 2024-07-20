#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author  : mystic
# @date    : 2018/6/3 18:57
"""
    SVD(Singular Value Decomposition),奇异值分解
优点: 简化数据, 去除噪声, 提高算法的结果
缺点: 数据的转换可能难以理解
适用数据类型: 数值型数据
"""

from numpy import corrcoef, eye, linalg, logical_and, mat, nonzero, shape, zeros


def load_external_data():
    return [
        [0, 0, 0, 2, 2],
        [0, 0, 0, 3, 3],
        [0, 0, 0, 1, 1],
        [1, 1, 1, 0, 0],
        [2, 2, 2, 0, 0],
        [5, 5, 5, 0, 0],
        [1, 1, 1, 0, 0],
    ]


def load_external_data2():
    return [
        [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
        [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
        [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
        [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
        [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
        [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
        [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
        [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
        [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
        [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0],
    ]


def euclidean_similarity(input_a, input_b):
    return 1.0 / (1.0 + linalg.norm(input_a - input_b))


def pearson_correlation_coefficient(input_a, input_b):
    if len(input_a) < 3:
        return 1.0
    return 0.5 + 0.5 * corrcoef(input_a, input_b, rowvar=False)[0][1]


def cosine_similarity(input_a, input_b):
    num = float(input_a.T * input_b)
    denominator = linalg.norm(input_a) * linalg.norm(input_b)
    return 0.5 + 0.5 * (num / denominator)


def stand_estimate(data_mat, user, similarity_measure, item):
    n = shape(data_mat)[1]
    similarity_total = 0.0
    rate_similarity_total = 0.0
    for j in range(n):
        user_rating = data_mat[user, j]
        if user_rating == 0:
            continue
        overlap = nonzero(logical_and(data_mat[:, item].A > 0, data_mat[:, j].A > 0))[0]
        if len(overlap) == 0:
            similarity = 0
        else:
            similarity = similarity_measure(
                data_mat[overlap, item], data_mat[overlap, j]
            )
        print("the %d and %d similarity is: %f" % (item, j, similarity))
        similarity_total += similarity
        rate_similarity_total += similarity * user_rating
    if similarity_total == 0:
        return 0
    else:
        return rate_similarity_total / similarity_total


def svd_estimate(data_mat, user, similarity_measure, item):
    n = shape(data_mat)[1]
    similarity_total = 0.0
    rate_similarity_total = 0.0
    u, sigma, v_t = linalg.svd(data_mat)
    sig4 = mat(eye(4) * sigma[:4])  # arrange Sig4 into a diagonal matrix
    x_formed_items = data_mat.T * u[:, :4] * sig4.I  # create transformed items
    for j in range(n):
        user_rating = data_mat[user, j]
        if user_rating == 0 or j == item:
            continue
        similarity = similarity_measure(
            x_formed_items[item, :].T, x_formed_items[j, :].T
        )
        print("the %d and %d similarity is: %f" % (item, j, similarity))
        similarity_total += similarity
        rate_similarity_total += similarity * user_rating
    if similarity_total == 0:
        return 0
    else:
        return rate_similarity_total / similarity_total


def recommend(
    data_mat,
    user,
    n=3,
    similarity_measure=cosine_similarity,
    estimate_method=stand_estimate,
):
    unrated_items = nonzero(data_mat[user, :].A == 0)[1]  # find unrated items
    if len(unrated_items) == 0:
        return "you rated everything"
    item_scores = []
    for item in unrated_items:
        estimated_score = estimate_method(data_mat, user, similarity_measure, item)
        item_scores.append((item, estimated_score))
    return sorted(item_scores, key=lambda jj: jj[1], reverse=True)[:n]


def print_matrix(input_matrix, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(input_matrix[i, k]) > thresh:
                print(1)
            else:
                print(0)
        print("")


def image_compress(num_s_v=3, thresh=0.8):
    myl = []
    for line in open("resource/0_5.txt").readlines():
        new_row = []
        for i in range(32):
            new_row.append(int(line[i]))
        myl.append(new_row)
    my_mat = mat(myl)
    print("****original matrix******")
    print_matrix(my_mat, thresh)
    u, sigma, v_t = linalg.svd(my_mat)
    sig_recon = mat(zeros((num_s_v, num_s_v)))
    for k in range(num_s_v):  # construct diagonal matrix from vector
        sig_recon[k, k] = sigma[k]
    recon_mat = u[:, :num_s_v] * sig_recon * v_t[:num_s_v, :]
    print("****reconstructed matrix using %d singular values******" % num_s_v)
    print_matrix(recon_mat, thresh)


if __name__ == "__main__":
    pass
