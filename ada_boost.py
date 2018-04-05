#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by PyCharm
# @author  : mystic
# @date    : 2018/4/2 19:10

"""
    AdaBoost
优点:
    泛化错误率低,易编码,可以应用在大部分分类器上,无参数调整
缺点:
    对离群点敏感
适用数据类型:
    数值型和标称型数据

    一般流程
1.收集数据:
    可以使用任意方法
2.准备数据:
    依赖于所使用的弱分类器类型,本章使用的是单层决策树,这种分类器可以处理任何数据类型
    当然也可以使用任意分类器作为弱分类器,作为弱分类器,简单分类器的效果更好
3.分析数据:
    可以使用任意方法
4.训练算法:
    AdaBoost的大部分时间都用在训练上,分类器将多次在同一条数据集上训练弱分类器
5.测试算法:
    计算分类的错误率
6.使用算法:
    同SVM一样,AdaBoost预测两个类别中的一个
    如果应用到多个类别,需要像多类SVM的做法一样,对AdaBoost进行修改
"""
from numpy import mat, ones, shape, zeros, inf, log, multiply, exp, sign


def load_simple_data():
    data_mat = mat([[1., 2.1],
                    [2., 1.1],
                    [1.3, 1.],
                    [1., 1.],
                    [2., 1.]])
    class_label_list = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_mat, class_label_list


def stump_classify(data_mat, dimension, thresh_val, thresh_unequal):
    return_arr = ones((shape(data_mat)[0], 1))
    if thresh_unequal == 'lt':
        return_arr[data_mat[:, dimension] <= thresh_val] = -1.0
    else:
        return_arr[data_mat[:, dimension] > thresh_val] = 1.0
    return return_arr


def build_stump(data_list, class_label_list, d):
    """
        build best decision stump
    :param data_list:
    :param class_label_list:
    :param d:
    :return:
    """
    data_mat = mat(data_list)
    label_mat = mat(class_label_list).T
    m, n = shape(data_mat)
    num_steps = 10
    # create empty dict
    best_stump = {}
    best_class_estimation = mat(zeros((m, 1)))
    # init error sum, to +infinity
    min_err = inf
    # loop over all dimensions
    for i in range(n):
        range_min = data_mat[:, i].min()
        range_max = data_mat[:, i].max()
        step_size = (range_max - range_min) / num_steps
        # loop over all range in current dimension
        for j in range(-1, int(num_steps) + 1):
            for unequal in ['lt', 'gt']:
                thresh_val = (range_min + float(j) * step_size)
                predicted_val_arr = stump_classify(data_mat, i, thresh_val, unequal)
                err_mat = mat(ones((m, 1)))
                err_mat[predicted_val_arr == label_mat] = 0
                # calc the weighted error rate
                weighted_err = d.T * err_mat
                # print("split: dimension %d, thresh %.2f, thresh unequal: %s, the weighted error is %.3f"
                #       % (i, thresh_val, unequal, weighted_err))
                if weighted_err < min_err:
                    min_err = weighted_err
                    best_class_estimation = predicted_val_arr.copy()
                    best_stump['dimension'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['unequal'] = unequal
    return best_stump, min_err, best_class_estimation


def ada_boost_train_decision_stump(data_list, class_label_list, iterator=40):
    weak_class_list = []
    m = shape(data_list)[0]
    # init D to all equal
    d = mat(ones((m, 1)) / m)
    aggregate_class_estimation = mat(zeros((m, 1)))
    for i in range(iterator):
        # build stump
        best_stump, err, class_estimation = build_stump(data_list, class_label_list, d)
        print('D: ', d.T)
        # calc alpha, throw in max(error,eps) to account for error=0
        # meanwhile, transfer list type to numerical type
        alpha = float(0.5 * log((1.0 - err) / max(err, 1e-16)))
        best_stump['alpha'] = alpha
        # store stump params in list
        weak_class_list.append(best_stump)
        print('class estimation: ', class_estimation.T)
        exponent = multiply(-1 * alpha * mat(class_label_list).T, class_estimation)
        # calc new d for next iteration
        d = multiply(d, exp(exponent))
        d = d / d.sum()
        aggregate_class_estimation += alpha * class_estimation
        print('aggregate class estimation: ', aggregate_class_estimation.T)
        aggregate_errors = multiply(sign(aggregate_class_estimation) != mat(class_label_list).T, ones((m, 1)))
        err_rate = aggregate_errors.sum() / m
        print('total error: %f\n' % err_rate)
        # calc training error of all classifiers, if this is 0 quit for loop early
        if err_rate == 0.0:
            break
    return weak_class_list


if __name__ == '__main__':
    data_matrix, labels = load_simple_data()
    # single decision stump
    # DD = mat(ones((5, 1)) / 5)
    # print(build_stump(data_matrix, labels, DD))
    print(ada_boost_train_decision_stump(data_matrix, labels, 9))
