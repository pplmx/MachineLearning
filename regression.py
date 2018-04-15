#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author  : mystic
# @date    : 2018/4/9 20:08
from numpy import mat, linalg, shape, eye, exp, zeros, mean, var


def load_data_set(filename):
    data_list = []
    label_list = []
    with open(filename) as fr:
        num_feature = len(fr.readline().split('\t'))
        for line in fr.readlines():
            line_list = []
            current_line = line.strip().split('\t')
            for i in range(num_feature - 1):
                line_list.append(float(current_line[i]))
            data_list.append(line_list)
            label_list.append(float(current_line[-1]))
    return data_list, label_list


def standard_regression(x_list, y_list):
    x_mat = mat(x_list)
    y_mat = mat(y_list).T
    x_t_x = x_mat.T * x_mat
    # determinant cannot be 0
    if linalg.det(x_t_x) == 0.0:
        print('This matrix is singular, cannot do inverse.')
        return
    ws = x_t_x.I * (x_mat.T * y_mat)
    # or invoke function in linalg, to solve unknown matrix
    # ws = linalg.solve(x_t_x, x_mat.T * y_mat)
    return ws


def locally_weighed_linear_regression(test_point, x_list, y_list, k=1.0):
    x_mat = mat(x_list)
    y_mat = mat(y_list).T
    m = shape(x_mat)[0]
    # create m*m square matrix
    # Return a 2-D array with ones on the diagonal and zeros elsewhere.
    weight_mat = mat(eye(m))
    for j in range(m):
        differ_mat = test_point - x_mat[j, :]
        weight_mat[j, j] = exp(differ_mat * differ_mat.T / (-2.0 * k ** 2))
    x_t_x = x_mat.T * (weight_mat * x_mat)
    if linalg.det(x_t_x) == 0.0:
        print('This matrix is singular, cannot do inverse.')
        return
    ws = x_t_x.I * (x_mat.T * (weight_mat * y_mat))
    return test_point * ws


def lwlr_test(test_list, x_list, y_list, k=1.0):
    m = shape(test_list)[0]
    y_hat = zeros(m)
    for i in range(m):
        y_hat[i] = locally_weighed_linear_regression(test_list[i], x_list, y_list, k)
    return y_hat


def ridge_regression(x_mat, y_mat, lambda_=0.2):
    x_t_x = x_mat.T * x_mat
    denominator = x_t_x + eye(shape(x_mat)[1]) * lambda_
    if linalg.det(denominator) == 0.0:
        print('This matrix is singular, cannot do inverse.')
        return
    ws = denominator.I * (x_mat.T * y_mat)
    return ws


def ridge_test(x_list, y_list):
    x_mat = mat(x_list)
    y_mat = mat(y_list).T
    # Compute the arithmetic mean along the specified axis.
    y_mean = mean(y_mat, 0)
    y_mat = y_mat - y_mean
    x_mean = mean(x_mat, 0)
    # Compute the variance along the specified axis.
    x_var = var(x_mat, 0)
    x_mat = (x_mat - x_mean) / x_var
    num_test = 30
    w_mat = zeros((num_test, shape(x_mat)[1]))
    for i in range(num_test):
        ws = ridge_regression(x_mat, y_mat, exp(i - 10))
        w_mat[i, :] = ws.T
    return w_mat


def plot_standard(x_list, y_list, ws):
    import matplotlib.pyplot as plt
    x_mat = mat(x_list)
    y_mat = mat(y_list).T
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_mat[:, 1].flatten().A[0], y_mat[:, 0].flatten().A[0])

    x_copy = x_mat.copy()
    x_copy.sort(0)
    y_hat = x_copy * ws
    ax.plot(x_copy[:, 1], y_hat)

    plt.show()


def plot_lwlr(x_list, y_list, y_hat):
    import matplotlib.pyplot as plt
    # sort x_list
    x_mat = mat(x_list)
    sort_idx = x_mat[:, 1].argsort(0)
    x_sort = x_mat[sort_idx][:, 0, :]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_sort[:, 1], y_hat[sort_idx])
    ax.scatter(x_mat[:, 1].flatten().A[0], mat(y_list).T.flatten().A[0], s=2, c='red')
    plt.show()


def plot_ridge(w_mat):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(w_mat)
    plt.show()


if __name__ == '__main__':
    # x_list_, y_list_ = load_data_set('resource/ex0.txt')
    # ws_ = standard_regression(x_list_, y_list_)
    # plot_standard(x_list_, y_list_, ws_)
    # y_hat_ = lwlr_test(x_list_, x_list_, y_list_, 0.003)
    # plot_lwlr(x_list_, y_list_, y_hat_)
    ab_x, ab_y = load_data_set('resource/abalone.txt')
    w_mat_ = ridge_test(ab_x, ab_y)
    plot_ridge(w_mat_)
