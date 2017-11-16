#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by PyCharm
# @author  : mystic
# @date    : 2017/11/16 15:59
"""
    decision tree
    核心思想:
        一种树结构（可以是二叉树或非二叉树）
        其每个非叶节点表示一个特征属性上的测试，
        每个分支代表这个特征属性在某个值域上的输出，
        而每个叶节点存放一个类别
    优点:
        计算复杂度不高,输出结果易于理解,对中间值缺失不敏感,可以处理不相关特征数据
    缺点：
        可能会产生过度匹配问题
    适用数据范围：
        数值型和标称型
"""
from math import log


def calc_shannon_entropy(data_set):
    num_entries = len(data_set)
    label_counts = {}
    for feat_vector in data_set:
        current_label = feat_vector[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_entropy = 0.0
    for key in label_counts:
        # 计算该分类的概率
        probability = label_counts[key] / num_entries
        # 通过循环,将各分类的信息期望值相加
        shannon_entropy -= probability * log(probability, 2)
    # 返回香农熵
    return shannon_entropy


def create_data_set():
    data_set = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


if __name__ == '__main__':
    my_data_set, my_labels = create_data_set()
    print(my_data_set)
    print(my_labels)
    my_shannon_entropy = calc_shannon_entropy(my_data_set)
    print(my_shannon_entropy)
