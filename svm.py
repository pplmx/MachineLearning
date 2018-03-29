#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by PyCharm
# @author  : mystic
# @date    : 2018/3/20 20:39

"""
    SVM: Support Vector Machine - 支持向量机
    优点:
        泛化错误率低,计算开销不大,结果易解释
    缺点:
        对参数调节和核函数的选择敏感,原始分类器不加修改仅适用于处理二类问题
    适用数据范围:
        数值型和标称型数据
    SVM的一般流程
        [1]收集数据: 可以使用任意方法
        [2]准备数据: 需要数值型数据
        [3]分析数据: 有助于可视化分隔超平面
        [4]训练算法: SVM的大部分时间都源自训练,该训练主要实现两个参数的调优
        [5]测试算法: 十分简单的计算过程就可以实现
        [6]使用算法: 几乎所有分类问题都可以使用SVM,值得一提的是,SVM本身是一个二类分类器
                    对多类问题应用,SVM需要对代码做一些修改
"""
from numpy import random, mat, shape, zeros, multiply, nonzero, exp, sign


def load_data_set(file_name):
    data_list = []
    label_list = []
    with open(file_name) as fr:
        for line in fr.readlines():
            line_list = line.strip().split('\t')
            data_list.append([float(line_list[0]), float(line_list[1])])
            label_list.append(float(line_list[2]))
    return data_list, label_list


def select_j_rand(i, m):
    """
        we want to select any J not equal to i
    :param i:
    :param m:
    :return:
    """
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clip_alpha(aj, high, low):
    if aj > high:
        aj = high
    if low > aj:
        aj = low
    return aj


def simple_smo(data_list, class_label_list, constant, tolerate, max_loop):
    """
        simple Sequential Minimal Optimization
    :param data_list: 数据集
    :param class_label_list: 分类标签
    :param constant: 常量
    :param tolerate: 容错率
    :param max_loop: 退出前的最大循环次数
    :return:
    """
    data_mat = mat(data_list)
    label_mat = mat(class_label_list).transpose()
    b = 0
    m, n = shape(data_mat)
    alpha_mat = mat(zeros((m, 1)))
    loop = 0
    while loop < max_loop:
        alpha_pairs_changed = 0
        for i in range(m):
            f_x_i = float(multiply(alpha_mat, label_mat).T * (data_mat * data_mat[i, :].T)) + b
            err_i = f_x_i - float(label_mat[i])
            is_okay = ((label_mat[i] * err_i < -tolerate) and (alpha_mat[i] < constant)) or \
                      ((label_mat[i] * err_i > tolerate) and (alpha_mat[i] > 0))
            if is_okay:
                j = select_j_rand(i, m)
                f_x_j = float(multiply(alpha_mat, label_mat).T * (data_mat * data_mat[j, :].T)) + b
                err_j = f_x_j - float(label_mat[j])
                alpha_i_old = alpha_mat[i].copy()
                alpha_j_old = alpha_mat[j].copy()
                if label_mat[i] != label_mat[j]:
                    low = max(0, alpha_mat[j] - alpha_mat[j])
                    high = min(constant, constant + alpha_mat[j] - alpha_mat[i])
                else:
                    low = max(0, alpha_mat[j] + alpha_mat[j] - constant)
                    high = min(constant, alpha_mat[j] + alpha_mat[i])
                if low == high:
                    print('low == high')
                    continue
                eta = 2.0 * data_mat[i, :] * data_mat[j, :].T - (data_mat[i, :] * data_mat[i, :].T) - (
                        data_mat[j, :] * data_mat[j, :].T)
                if eta >= 0:
                    print('eta >= 0')
                    continue
                alpha_mat[j] -= label_mat[j] * (err_i - err_j) / eta
                alpha_mat[j] = clip_alpha(alpha_mat[j], high, low)
                if (abs(alpha_mat[j]) - alpha_j_old) < 0.00001:
                    print('j not moving enough')
                    continue
                alpha_mat[i] += label_mat[j] * label_mat[i] * (alpha_j_old - alpha_mat[j])
                b1 = b - err_i - label_mat[i] * (alpha_mat[i] - alpha_i_old) * data_mat[i, :] * data_mat[i, :].T - \
                     label_mat[j] * (alpha_mat[j] - alpha_j_old) * data_mat[i, :] * data_mat[j, :].T
                b2 = b - err_j - label_mat[i] * (alpha_mat[i] - alpha_i_old) * data_mat[i, :] * data_mat[j, :].T - \
                     label_mat[j] * (alpha_mat[j] - alpha_j_old) * data_mat[j, :] * data_mat[j, :].T
                if (0 < alpha_mat[i]) and (constant > alpha_mat[i]):
                    b = b1
                elif (0 < alpha_mat[j]) and (constant > alpha_mat[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_pairs_changed += 1
                print('loop: %d, i: %d, pairs changed: %d' % (loop, i, alpha_pairs_changed))
        if alpha_pairs_changed == 0:
            loop += 1
        else:
            loop = 0
        print('loop number: %d' % loop)
    return b, alpha_mat


class OptStruct:
    def __init__(self, input_data_mat, class_label_mat, constant, tolerate, k_tup):
        self.X = input_data_mat
        self.label_mat = class_label_mat
        self.C = constant
        self.tol = tolerate
        self.m = shape(input_data_mat)[0]
        self.alpha_mat = mat(zeros((self.m, 1)))
        self.b = 0
        self.err_cache = mat(zeros((self.m, 2)))
        self.K = mat(zeros((self.m, self.m)))


def calc_err_k(o_s, k):
    f_x_k = float(multiply(o_s.alpha_mat, o_s.label_mat).T
                  * o_s.K[:, k] + o_s.b)
    err_k = f_x_k - float(o_s.label_mat[k])
    return err_k


def select_j(i, o_s, err_i):
    max_k = -1
    max_delta_err = 0
    err_j = 0
    o_s.err_cache[i] = [1, err_i]
    valid_err_cache_list = nonzero(o_s.err_cache[:, 0].A)[0]
    if len(valid_err_cache_list) > 1:
        for k in valid_err_cache_list:
            if k == i:
                continue
            err_k = calc_err_k(o_s, k)
            delta_err = abs(err_i - err_k)
            if delta_err > max_delta_err:
                max_k = k
                max_delta_err = delta_err
                err_j = err_k
        return max_k, err_j
    else:
        j = select_j_rand(i, o_s.m)
        err_j = calc_err_k(o_s, j)
    return j, err_j


def update_err_k(o_s, k):
    err_k = calc_err_k(o_s, k)
    o_s.err_cache[k] = [1, err_k]


def inner_l(i, o_s):
    err_i = calc_err_k(o_s, i)
    is_okay = ((o_s.label_mat[i] * err_i < -o_s.tol) and (o_s.alpha_mat[i] < o_s.C)) or \
              ((o_s.label_mat[i] * err_i > o_s.tol) and (o_s.alpha_mat[i] > 0))
    if is_okay:
        j, err_j = select_j(i, o_s, err_i)
        alpha_i_old = o_s.alpha_mat[i].copy()
        alpha_j_old = o_s.alpha_mat[j].copy()
        if o_s.label_mat[i] != o_s.label_mat[j]:
            low = max(0, o_s.alpha_mat[j] - o_s.alpha_mat[i])
            high = min(o_s.C, o_s.C + o_s.alpha_mat[j] - o_s.alpha_mat[i])
        else:
            low = max(0, o_s.alpha_mat[j] + o_s.alpha_mat[i] - o_s.C)
            high = min(o_s.C, o_s.alpha_mat[j] + o_s.alpha_mat[i])
        if low == high:
            print('low==high')
            return 0
        eta = 2.0 * o_s.K[i, j] - o_s.K[i, i] - o_s.K[j, j]
        if eta >= 0:
            print('eta >= 0')
            return 0
        o_s.alpha_mat[j] -= o_s.label_mat[j] * (err_i - err_j) / eta
        o_s.alpha_mat[j] = clip_alpha(o_s.alpha_mat[j], high, low)
        update_err_k(o_s, j)
        if abs(o_s.alpha_mat[j] - alpha_j_old) < 0.00001:
            print('j not moving enough')
            return 0
        o_s.alpha_mat[i] += o_s.label_mat[j] * o_s.label_mat[i] * (alpha_j_old - o_s.alpha_mat[j])
        update_err_k(o_s, i)

        b1_tmp1 = o_s.b - err_i - o_s.label_mat[i] * (o_s.alpha_mat[i] - alpha_i_old) * o_s.K[i, i]
        b1_tmp2 = o_s.label_mat[j] * (o_s.alpha_mat[j] - alpha_j_old) * o_s.K[i, j]
        b1 = b1_tmp1 - b1_tmp2

        b2_tmp1 = o_s.b - err_j - o_s.label_mat[i] * (o_s.alpha_mat[i] - alpha_i_old) * o_s.K[i, j]
        b2_tmp2 = o_s.label_mat[j] * (o_s.alpha_mat[j] - alpha_j_old) * o_s.K[j, j]
        b2 = b2_tmp1 - b2_tmp2

        if (0 < o_s.alpha_mat[i]) and (o_s.C > o_s.alpha_mat[i]):
            o_s.b = b1
        elif (0 < o_s.alpha_mat[j]) and (o_s.C > o_s.alpha_mat[j]):
            o_s.b = b2
        else:
            o_s.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def platt_smo(data_list, class_label_list, constant, tolerate, max_loop, k_tup=('lin', 0)):
    o_s = OptStruct(mat(data_list), mat(class_label_list).transpose(), constant, tolerate, k_tup)
    loop = 0
    entire_set = True
    alpha_pairs_changed = 0
    while (loop < max_loop) and ((alpha_pairs_changed > 0) or entire_set):
        alpha_pairs_changed = 0
        if entire_set:
            for i in range(o_s.m):
                alpha_pairs_changed += inner_l(i, o_s)
                print('fullSet, loop: %d, i: %d, pairs changed %d' % (loop, i, alpha_pairs_changed))
            loop += 1
        else:
            non_bound_is = nonzero((o_s.alpha_mat.A > 0) * (o_s.alpha_mat.A < constant))[0]
            for i in non_bound_is:
                alpha_pairs_changed += inner_l(i, o_s)
                print('non bound, loop: %d, i: %d, pairs changed %d' % (loop, i, alpha_pairs_changed))
            loop += 1
        if entire_set:
            entire_set = False
        elif alpha_pairs_changed == 0:
            entire_set = 0
        print('loop number: %d' % loop)
    return o_s.b, o_s.alpha_mat


def calc_ws(alpha_mat, data_list, class_label_list):
    x = mat(data_list)
    label_mat = mat(class_label_list)
    m, n = shape(x)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alpha_mat[i] * label_mat[i], x[i, :].T)
    return w


def kernel_translation(x, a, k_tup):
    m, n = shape(x)
    k = mat(zeros((m, 1)))
    if k_tup[0] == 'lin':
        k = x * a.T
    elif k_tup[0] == 'rbf':
        for j in range(m):
            delta_row = x[j, :] - a
            k[j] = delta_row * delta_row.T
        k = exp(k/(-1*k_tup[1]**2))
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return k


def test_rbf(k1 = 1.3):
    data_arr, label_arr = load_data_set('resource/testSetRBF.txt')
    b, alpha_matrix = platt_smo(data_arr, label_arr, 200, 0.0001, 10000, ('rbf', k1))
    data_matrix = mat(data_arr)
    label_matrix = mat(label_arr).transpose()
    sv_ind = nonzero(alpha_matrix.A > 0)[0]
    s_v_s = data_matrix[sv_ind]
    label_sv = label_matrix[sv_ind]
    print('There are %d Support Vectors' % shape(s_v_s)[0])
    m, n = shape(data_matrix)
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_translation(s_v_s, data_matrix[i, :], ('rbf', k1))
        predict = kernel_eval.T * multiply(label_sv, alpha_matrix[sv_ind]) + b
        if sign(predict) != sign(label_arr[i]):
            error_count += 1
    print('The training error rate is: %f' % (error_count/m))
    data_arr, label_arr = load_data_set('resource/testSetRBF2.txt')
    error_count = 0
    data_matrix = mat(data_arr)
    label_matrix = mat(label_arr).transpose()
    m, n = shape(data_matrix)
    for i in range(m):
        kernel_eval = kernel_translation(s_v_s, data_matrix[i, :], ('rbf', k1))
        predict = kernel_eval.T * multiply(label_sv, alpha_matrix[sv_ind]) + b
        if sign(predict) != sign(label_arr[i]):
            error_count += 1
    print('The test error rate is： %f' % (error_count/m))


if __name__ == "__main__":
    # data_arr, label_arr = load_data_set('resource/testSet1.txt')
    # bb, alphas = simple_smo(data_arr, label_arr, 0.6, 0.001, 40)
    # bb, alphas = platt_smo(data_arr, label_arr, 0.6, 0.001, 40)
    test_rbf()
