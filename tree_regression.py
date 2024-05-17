#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author  : mystic
# @date    : 2018/4/19 20:06

"""
    树回归
优点: 可以对复杂和非线性的数据建模
缺点: 结果不易理解
适用数据类型: 数值型和标称型数据
    树回归的一般方法
1.收集数据: 采用任一方法收集数据
2.准备数据: 需要数值型的数据,标称型数据应该映射成二值型数据
3.分析数据: 绘出数据的二维可视化显示结果,以字典方式生成树
4.训练算法: 大部分时间都花费在叶节点树模型的构建上
5.测试算法: 使用测试数据上的R平方值来分析模型的效果
6.使用算法: 使用训练出的树做预测,预测结果还可以用来做很多事情
"""

from numpy import nonzero, mean, var, shape, inf, mat, power, ones, zeros
from scipy import linalg


def load_data_set(filename):  # general function to parse tab -delimited floats
    data_list = []  # assume last column is target value
    with open(filename) as fr:
        for line in fr.readlines():
            cur_line = line.strip().split("\t")
            # TODO the return type has changed in python3
            # before change: map(float, cur_line)
            # after change: list(map(float, cur_line))
            flt_line = list(map(float, cur_line))  # map all elements to float()
            data_list.append(flt_line)
        return data_list


def bin_split_data_set(data_mat, feature, value):
    # TODO
    # before change: will get index 0 of array
    # after change: not get index 0 of array
    mat_0 = data_mat[nonzero(data_mat[:, feature] > value)[0], :]
    mat_1 = data_mat[nonzero(data_mat[:, feature] <= value)[0], :]
    return mat_0, mat_1


def regression_leaf(data_mat):  # returns the value used for each leaf
    return mean(data_mat[:, -1])


def regression_err(data_mat):
    return var(data_mat[:, -1]) * shape(data_mat)[0]


def linear_solve(data_set):
    """
    helper function used in two places
    :param data_set:
    :return:
    """
    m, n = shape(data_set)
    x = mat(ones((m, n)))
    # create a copy of data with 1 in 0th position
    # y = mat(ones((m, 1)))
    x[:, 1:n] = data_set[:, 0 : n - 1]
    y = data_set[:, -1]
    x_t_x = x.T * x
    if linalg.det(x_t_x) == 0:
        raise NameError(
            "This matrix is singular, cannot do inverse,\n\
                try increasing the second value of ops"
        )
    ws = x_t_x.I * (x.T * y)
    return ws, x, y


def model_leaf(data_set):
    ws, x, y = linear_solve(data_set)
    return ws


def model_err(data_set):
    ws, x, y = linear_solve(data_set)
    y_hat = x * ws
    return sum(power(y - y_hat, 2))


def choose_best_split(
    data_mat, leaf_type=regression_leaf, err_type=regression_err, ops=(1, 4)
):
    tolerance_split = ops[0]
    tolerance_n = ops[1]
    # if all the target variables are the same value: quit and return value
    if len(set(data_mat[:, -1].T.tolist()[0])) == 1:  # exit condition 1
        return None, leaf_type(data_mat)
    m, n = shape(data_mat)
    # the choice of the best feature is driven by Reduction in RSS error from mean
    split = err_type(data_mat)
    best_split = inf
    best_idx = 0
    best_val = 0
    for feature_idx in range(n - 1):
        # TODO
        # before change: data_set[:, feature_idx]
        # after change: data_set[:, feature_idx].T.tolist()[0]
        for split_val in set(data_mat[:, feature_idx].T.tolist()[0]):
            mat_0, mat_1 = bin_split_data_set(data_mat, feature_idx, split_val)
            if (shape(mat_0)[0] < tolerance_n) or (shape(mat_1)[0] < tolerance_n):
                continue
            new_s = err_type(mat_0) + err_type(mat_1)
            if new_s < best_split:
                best_idx = feature_idx
                best_val = split_val
                best_split = new_s
    # if the decrease (split-best_split) is less than a threshold don't do the split
    if (split - best_split) < tolerance_split:  # exit condition 2
        return None, leaf_type(data_mat)
    mat_0, mat_1 = bin_split_data_set(data_mat, best_idx, best_val)
    if (shape(mat_0)[0] < tolerance_n) or (
        shape(mat_1)[0] < tolerance_n
    ):  # exit condition 3
        return None, leaf_type(data_mat)
    # returns the best feature to split on
    # and the value used for that split
    return best_idx, best_val


def create_tree(
    data_mat, leaf_type=regression_leaf, err_type=regression_err, ops=(1, 4)
):
    # assume data_set is NumPy Mat so we can array filtering
    # choose the best split
    feature, val = choose_best_split(data_mat, leaf_type, err_type, ops)
    # if the splitting hit a stop condition return val
    if feature is None:
        return val
    tree = {"split_idx": feature, "split_val": val}
    left_set, right_set = bin_split_data_set(data_mat, feature, val)
    tree["left"] = create_tree(left_set, leaf_type, err_type, ops)
    tree["right"] = create_tree(right_set, leaf_type, err_type, ops)
    return tree


def is_tree(obj):
    return type(obj).__name__ == "dict"


def get_mean(tree):
    if is_tree(tree["right"]):
        tree["right"] = get_mean(tree["right"])
    if is_tree(tree["left"]):
        tree["left"] = get_mean(tree["left"])
    return (tree["left"] + tree["right"]) / 2.0


def prune(tree, test_data):
    # if we have no test data collapse the tree
    if shape(test_data)[0] == 0:
        return get_mean(tree)
    left_set, right_set = bin_split_data_set(
        test_data, tree["split_idx"], tree["split_val"]
    )
    if is_tree(tree["right"]) or is_tree(tree["left"]):
        if is_tree(tree["left"]):
            tree["left"] = prune(tree["left"], left_set)
        if is_tree(tree["right"]):
            tree["right"] = prune(tree["right"], right_set)
    else:
        # if they are now both leafs, see if we can merge them
        err_no_merge = sum(power(left_set[:, 1] - tree["left"], 2)) + sum(
            power(right_set[:, 1] - tree["right"], 2)
        )
        tree_mean = (tree["left"] + tree["right"]) / 2.0
        err_merge = sum(power(test_data[:, 1] - tree_mean, 2))
        if err_merge < err_no_merge:
            print("Merging")
            return tree_mean
        else:
            return tree
    return tree


def regression_tree_evaluation(model):
    return float(model)


def model_tree_evaluation(model, input_data):
    n = shape(input_data)[1]
    x = mat(ones((1, n + 1)))
    x[:, 1 : n + 1] = input_data
    return float(x * model)


def tree_forecast(tree, input_data, model_eval=regression_tree_evaluation):
    if not is_tree(tree):
        return model_eval(tree)
    if input_data[tree["split_idx"]] > tree["split_val"]:
        if is_tree(tree["left"]):
            return tree_forecast(tree["left"], input_data, model_eval)
        else:
            return model_eval(tree["left"])
    else:
        if is_tree(tree["right"]):
            return tree_forecast(tree["right"], input_data, model_eval)
        else:
            return model_eval(tree["right"])


def create_forecast(tree, test_data, model_eval=regression_tree_evaluation):
    m = len(test_data)
    y_hat = mat(zeros((m, 1)))
    for i in range(m):
        y_hat[i, 0] = tree_forecast(tree, mat(test_data[i]), model_eval)
    return y_hat


if __name__ == "__main__":
    # data_list_ = load_data_set('resource/ex2.txt')
    # data_mat_ = mat(data_list_)
    # tree_ = create_tree(data_mat_, ops=(0, 1))
    #
    # test_data_list_ = load_data_set('resource/ex2test.txt')
    # test_data_mat_ = mat(test_data_list_)
    # pruned_tree_ = prune(tree_, test_data_mat_)
    # json_ = json.dumps(pruned_tree_, indent=4)
    #
    # print(json_)

    data_list_ = load_data_set("resource/exp2.txt")
    data_mat_ = mat(data_list_)
    tree_ = create_tree(data_mat_, model_leaf, model_err, ops=(1, 10))
    print(tree_)
