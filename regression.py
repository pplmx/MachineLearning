#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author  : mystic
# @date    : 2018/4/9 20:08
from numpy import mat, linalg


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


def stand_regression(x_list, y_list):
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


def plot(x_list, y_list, ws):
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


if __name__ == '__main__':
    x_list_, y_list_ = load_data_set('resource/ex0.txt')
    ws_ = stand_regression(x_list_, y_list_)
    print(ws_)
    plot(x_list_, y_list_, ws_)
