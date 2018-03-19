#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by PyCharm
# @author  : mystic
# @date    : 2/13/2018 16:38

"""
Logistic回归的一般过程
    (1)收集数据: 采用任意方法收集数据
    (2)准备数据: 由于需要进行距离计算,因此需求数据类型为数值型.另外,结构化数据格式则最佳
    (3)分析数据: 采用任意方法对数据进行分析
    (4)训练算法: 大部分时间将用于训练,训练的目的是为了找到最佳的分类回归系数
    (5)测试算法: 一旦训练步骤完成,分类将会很快
    (6)使用算法:
                首先,我们需要输入一些数据,并将其转化成对应的结构化数据;
                接着,基于训练好的回归系数就可以对这些数值进行简单的回归计算,判定它们属于哪个类别;
                在这之后,我么就可以在输出的类别上做一些其他分析工作
优点:
    计算代价不高,已于理解和实现
缺点:
    容易欠拟合,分类精度可能不高
使用数据类型:
    数值型和标称型数据
"""

from numpy import shape, exp, mat, ones, array, arange, random


def load_data_set():
    data_list = []
    label_list = []
    fr = open('resource/testSet.txt')
    for line in fr.readlines():
        # 去除首尾空白字符,并以空白字符分割
        line_list = line.strip().split()
        data_list.append([1.0, float(line_list[0]), float(line_list[1])])
        label_list.append(int(line_list[2]))
    return data_list, label_list


def sigmoid(x):
    return 1.0 / (1 + exp(-x))


def gradient_ascent(data_list, class_label_list):
    data_mat = mat(data_list)
    label_mat = mat(class_label_list).transpose()
    m, n = shape(data_mat)
    alpha = 0.001
    max_cycles = 500
    weight_arr = ones((n, 1))
    for k in range(max_cycles):
        h = sigmoid(data_mat * weight_arr)
        error = (label_mat - h)
        # 加上一个矩阵后,返回值类型为matrix
        weight_arr = weight_arr + alpha * data_mat.transpose() * error
    return array(weight_arr)


def stochastic_gradient_ascent(data_list, class_label_list):
    data_arr = array(data_list)
    m, n = shape(data_arr)
    alpha = 0.01
    weight_arr = ones(n)
    for i in range(m):
        h = sigmoid(sum(data_arr[i] * weight_arr))
        error = class_label_list[i] - h
        weight_arr = weight_arr + alpha * error * data_arr[i]
    return weight_arr


def improved_stochastic_gradient_ascent(data_list, class_label_list, iterator_times=150):
    data_arr = array(data_list)
    m, n = shape(data_arr)
    weight_arr = ones(n)
    for j in range(iterator_times):
        data_idx = list(range(m))
        for i in range(m):
            # 每次迭代,调整alpha
            alpha = 4 / (1.0 + j + i) + 0.01
            random_idx = int(random.uniform(0, len(data_idx)))
            h = sigmoid(sum(data_arr[random_idx] * weight_arr))
            error = class_label_list[random_idx] - h
            weight_arr = weight_arr + alpha * error * data_arr[random_idx]
            del (data_idx[random_idx])
    return weight_arr


def classify_vector(input_x, weight_arr):
    probability = sigmoid(sum(input_x * weight_arr))
    if probability > 0.5:
        return 1.0
    else:
        return 0.0


def colic_test():
    fr_train = open('resource/horseColicTraining.txt')
    fr_test = open('resource/horseColicTest.txt')
    training_set = []
    training_labels = []
    for line in fr_train.readlines():
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        training_set.append(line_arr)
        training_labels.append(float(curr_line[21]))
    train_weight_arr = improved_stochastic_gradient_ascent(array(training_set), training_labels, 500)
    error_count = 0
    num_test_vec = 0.0
    for line in fr_test.readlines():
        num_test_vec += 1.0
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        if int(classify_vector(array(line_arr), train_weight_arr)) != int(curr_line[21]):
            error_count += 1
    error_rate = error_count / num_test_vec
    print('The error rate of this test is: %f' % error_rate)
    return error_rate


def multi_test():
    num_tests = 10
    error_sum = 0.0
    for k in range(num_tests):
        error_sum += colic_test()
    print('after %d iterations the average error rate is: %f' % (num_tests, error_sum / num_tests))


def plot_best_fit(weight_arr):
    import matplotlib.pyplot as plt
    data_list, class_label_list = load_data_set()
    data_arr = array(data_list)
    n = shape(data_arr)[0]
    x_cord1_list = []
    y_cord1_list = []
    x_cord2_list = []
    y_cord2_list = []
    for i in range(n):
        if int(class_label_list[i]) == 1:
            x_cord1_list.append(data_arr[i, 1])
            y_cord1_list.append(data_arr[i, 2])
        else:
            x_cord2_list.append(data_arr[i, 1])
            y_cord2_list.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1_list, y_cord1_list, s=30, c='red', marker='s')
    ax.scatter(x_cord2_list, y_cord2_list, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    # 最佳拟合直线
    y = (-weight_arr[0] - weight_arr[1] * x) / weight_arr[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == '__main__':
    result = load_data_set()
    # 梯度上升
    # weights = gradient_ascent(result[0], result[1])
    # 随机梯度上升
    # weights = stochastic_gradient_ascent(result[0], result[1])
    # 改进的随机梯度上升
    # weights = improved_stochastic_gradient_ascent(result[0], result[1])
    # plot_best_fit(weights)
    multi_test()
