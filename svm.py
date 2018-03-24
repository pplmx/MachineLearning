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
from numpy import random, mat, shape, zeros, multiply


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
            e_i = f_x_i - float(label_mat[i])
            is_okay = ((label_mat[i] * e_i < -tolerate) and (alpha_mat[i] < constant)) or \
                      ((label_mat[i] * e_i > tolerate) and (alpha_mat[i] > 0))
            if is_okay:
                j = select_j_rand(i, m)
                f_x_j = float(multiply(alpha_mat, label_mat).T * (data_mat * data_mat[j, :].T)) + b
                e_j = f_x_j - float(label_mat[j])
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
                alpha_mat[j] -= label_mat[j] * (e_i - e_j) / eta
                alpha_mat[j] = clip_alpha(alpha_mat[j], high, low)
                if (abs(alpha_mat[j]) - alpha_j_old) < 0.00001:
                    print('j not moving enough')
                    continue
                alpha_mat[i] += label_mat[j] * label_mat[i] * (alpha_j_old - alpha_mat[j])
                b1 = b - e_i - label_mat[i] * (alpha_mat[i] - alpha_i_old) * data_mat[i, :] * data_mat[i, :].T - \
                     label_mat[j] * (alpha_mat[j] - alpha_j_old) * data_mat[i, :] * data_mat[j, :].T
                b2 = b - e_j - label_mat[i] * (alpha_mat[i] - alpha_i_old) * data_mat[i, :] * data_mat[j, :].T - \
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


if __name__ == "__main__":
    data_arr, label_arr = load_data_set('resource/testSet1.txt')
    bb, alphas = simple_smo(data_arr, label_arr, 0.6, 0.001, 40)
    print(bb)
    print(alphas[alphas > 0])
    print(shape(alphas[alphas > 0]))

