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

from numpy import array, exp, inf, log, mat, multiply, ones, shape, sign, zeros


def load_simple_data():
    data_mat = mat([[1.0, 2.1], [2.0, 1.1], [1.3, 1.0], [1.0, 1.0], [2.0, 1.0]])
    class_label_list = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_mat, class_label_list


def stump_classify(data_mat, dimension, thresh_val, thresh_unequal):
    return_arr = ones((shape(data_mat)[0], 1))
    if thresh_unequal == "lt":
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
            for unequal in ["lt", "gt"]:
                thresh_val = range_min + float(j) * step_size
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
                    best_stump["dimension"] = i
                    best_stump["thresh"] = thresh_val
                    best_stump["unequal"] = unequal
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
        # print('D: ', d.T)
        # calc alpha, throw in max(error,eps) to account for error=0
        # meanwhile, transfer list type to numerical type
        alpha = float(0.5 * log((1.0 - err) / max(err, 1e-16)))
        best_stump["alpha"] = alpha
        # store stump params in list
        weak_class_list.append(best_stump)
        # print('class estimation: ', class_estimation.T)
        exponent = multiply(-1 * alpha * mat(class_label_list).T, class_estimation)
        # calc new d for next iteration
        d = multiply(d, exp(exponent))
        d = d / d.sum()
        aggregate_class_estimation += alpha * class_estimation
        # print('aggregate class estimation: ', aggregate_class_estimation.T)
        aggregate_errors = multiply(
            sign(aggregate_class_estimation) != mat(class_label_list).T, ones((m, 1))
        )
        err_rate = aggregate_errors.sum() / m
        # print('total error: %f\n' % err_rate)
        # calc training error of all classifiers, if this is 0 quit for loop early
        if err_rate == 0.0:
            break
    return weak_class_list, aggregate_class_estimation


def ada_classify(data2class, classifier_list):
    data_mat = mat(data2class)
    m = shape(data_mat)[0]
    aggregate_class_estimation = mat(zeros((m, 1)))
    for i in range(len(classifier_list)):
        best_stump = classifier_list[i]
        class_estimation = stump_classify(
            data_mat,
            best_stump["dimension"],
            best_stump["thresh"],
            best_stump["unequal"],
        )
        aggregate_class_estimation += best_stump["alpha"] * class_estimation
        # print('aggregate class estimation: ', aggregate_class_estimation)
    return sign(aggregate_class_estimation)


def load_data_set(filename):
    data_list = []
    label_list = []
    with open(filename) as fr:
        num_feature = len(fr.readline().split("\t"))
        for line in fr.readlines():
            line_list = []
            current_line = line.strip().split("\t")
            for i in range(num_feature - 1):
                line_list.append(float(current_line[i]))
            data_list.append(line_list)
            label_list.append(float(current_line[-1]))
    return data_list, label_list


def plot_roc(prediction_strength, class_label_list):
    import matplotlib.pyplot as plt

    cursor = (1.0, 1.0)
    y_sum = 0.0
    num_pos_class = sum(array(class_label_list) == 1.0)
    y_step = 1 / float(num_pos_class)
    x_step = 1 / float(len(class_label_list) - num_pos_class)
    # get sorted index, it's reverse
    sorted_indices = prediction_strength.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # loop through all the values, drawing a line segment at each point
    for index in sorted_indices.tolist()[0]:
        if class_label_list[index] == 1.0:
            del_x = 0
            del_y = y_step
        else:
            del_x = x_step
            del_y = 0
            y_sum += cursor[1]
        # draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cursor[0], cursor[0] - del_x], [cursor[1], cursor[1] - del_y], c="b")
        cursor = (cursor[0] - del_x, cursor[1] - del_y)
    ax.plot([0, 1], [0, 1], "b--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve for AdaBoost horse colic detection system")
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("the Area Under the Curve is: ", y_sum * x_step)


if __name__ == "__main__":
    # data_list_, label_list_ = load_simple_data()
    data_list_, label_list_ = load_data_set("resource/horseColicTraining2.txt")
    # single decision stump
    # DD = mat(ones((5, 1)) / 5)
    # print(build_stump(data_list_, label_list_, DD))
    classifier_list_, aggregate_class_estimation_ = ada_boost_train_decision_stump(
        data_list_, label_list_, 10
    )
    test_data_list_, test_label_list_ = load_data_set("resource/horseColicTest2.txt")
    prediction_ = ada_classify(test_data_list_, classifier_list_)
    err_mat_ = mat(ones((shape(prediction_)[0], 1)))
    err_rate_ = err_mat_[prediction_ != mat(test_label_list_).T].sum()
    print(err_rate_)
    plot_roc(aggregate_class_estimation_.T, label_list_)
