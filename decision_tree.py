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
import operator
import pickle
from collections import Counter
from math import log

from tree_plotter import create_plot


def create_data_set():
    """
        数据集:
            1.必须是一种由列元素组成的列表,而且所有列表元素均具有相同的数据长度
            2.数据的最后一列或者每一个实例的最后一个元素是当前实例的类别标签
    :return:
    """
    data_set = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def calc_shannon_entropy(data_set):
    num_entries = len(data_set)
    label_counts = {}
    # 对各类别出现的次数,进行统计
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


def calc_shannon_entropy2(data_set):
    """
        通过列表推导式,及Counter,实现香农熵的计算
    :param data_set:
    :return:
    """
    # 取出'yes','yes','no'等数据放到数组中
    class_count = [sample[-1] for sample in data_set]
    # 获取数据集长度
    length = len(data_set)
    # 对'yes','no'等各类别出现的次数,进行统计
    class_count = Counter(class_count)
    shannon_entropy = 0.
    # 计算香农熵
    for times in class_count.values():
        shannon_entropy -= times / length * log(times / length, 2)
    return shannon_entropy


def split_data_set(data_set, axis, value):
    """
        划分数据集
    :param data_set: 待划分的数据集
    :param axis: 划分数据集的特征
    :param value: 需要返回的特征的值
    :return:
    """
    """
        append和extend的区别
        >>> a = [1,2,3]
        >>> b = [4,5,6]
        >>> a.append(b)
        >>> a
        [1, 2, 3, [4, 5, 6]]
        >>> a.extend(b)
        >>> a
        [1, 2, 3, [4, 5, 6], 4, 5, 6]
    """
    divided_data_set = []
    for feature_vector in data_set:
        # if true,就将该值remove,同时添加进divided_data_set
        if feature_vector[axis] == value:
            reduced_feature_vector = feature_vector[:axis]
            reduced_feature_vector.extend(feature_vector[axis + 1:])
            divided_data_set.append(reduced_feature_vector)
    return divided_data_set


def choose_best_feature2split(data_set):
    # 获取特征值的数量
    num_features = len(data_set[0]) - 1
    # 计算原始香农熵
    base_entropy = calc_shannon_entropy2(data_set)
    # 最佳信息增益
    best_info_gain = 0.
    # 最佳特征值的位置索引
    best_feature = -1
    for i in range(num_features):
        # 创建唯一的分类标签列表
        feature_list = [example[i] for example in data_set]
        unique_vals = set(feature_list)
        # 计算每种划分方式的信息熵
        new_entropy = 0.
        for value in unique_vals:
            sub_data_set = split_data_set(data_set, i, value)
            probability = len(sub_data_set) / len(data_set)
            new_entropy += probability * calc_shannon_entropy2(sub_data_set)
        # 计算最好的信息增益
        # print('原始信息熵为%f' % base_entropy)
        # print('新的信息熵为%f' % new_entropy)
        info_gain = base_entropy - new_entropy
        # print('按照第%d个特征属性划分,信息增益为%f' % (i, info_gain))
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    # print('故最佳特征属性的索引为%d' % best_feature)
    return best_feature


def majority_counter(class_list):
    """
        如果数据集已经处理了所有属性,但是类标签依然不是唯一的,我们需要决定如何定义该叶子节点
        此时,我们采用多数表决的方式,决定该叶子节点的分类
    :param class_list:
    :return:
    """
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def classify(input_tree, feature_labels, test_vector):
    first_str = list(input_tree.keys())[0]
    second_dict = input_tree[first_str]
    # 将标签字符串转换为索引
    feature_index = feature_labels.index(first_str)
    class_label = None
    for key in second_dict.keys():
        if test_vector[feature_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feature_labels, test_vector)
            else:
                class_label = second_dict[key]
    return class_label


def store_tree(input_tree, filename):
    with open(filename, 'wb') as fw:
        # 0：ASCII协议，所序列化的对象使用可打印的ASCII码表示
        # 1：老式的二进制协议
        # 2：2.3版本引入的新二进制协议，较以前的更高效
        # 其中协议0和1兼容老版本的python
        pickle.dump(input_tree, fw, 0)


def grab_tree(filename):
    with open(filename, 'rb') as read:
        return pickle.load(read)


def create_tree(data_set, labels):
    # 获取类别列表
    class_list = [example[-1] for example in data_set]
    # 类别完全相同,则停止继续划分
    if class_list.count(class_list[0]) == len(class_list):
        # 若只有一类,则某个类别标签的数量,应该和它的数据长度相等
        return class_list[0]
    # 遍历完所有特征时,类别标签还是不唯一,则返回出现次数最多的类别
    if len(data_set[0]) == 1:
        return majority_counter(class_list)
    # 最佳特征属性的索引
    best_feature = choose_best_feature2split(data_set)
    # 最佳特征标记
    best_feature_label = labels[best_feature]
    # 创建字典,存储决策树
    my_tree = {best_feature_label: {}}
    del (labels[best_feature])
    # 获取该特征的所有的值
    feature_values = [example[best_feature] for example in data_set]
    unique_values = set(feature_values)
    for value in unique_values:
        sub_labels = labels[:]
        # 递归不断创建分支
        my_tree[best_feature_label][value] = create_tree(split_data_set(data_set, best_feature, value), sub_labels)
    return my_tree


if __name__ == '__main__':
    # my_data_set, my_labels = create_data_set()
    # print(my_data_set)
    # print(my_labels)
    # my_shannon_entropy = calc_shannon_entropy(my_data_set)
    # print(my_shannon_entropy)
    # print(calc_shannon_entropy2(my_data_set))
    # decision_tree = create_tree(my_data_set, my_labels)
    # store_tree(decision_tree, 'resource/classifier_storage.txt')
    # print(decision_tree)
    # new_tree = grab_tree('resource/classifier_storage.txt')
    # print('acquire tree from file:', new_tree)
    # # 因为my_labels已经在create_tree方法中被改变,故我们生成个新的
    # my_data_set, my_labels = create_data_set()
    # print(classify(decision_tree, my_labels, [1, 1]))
    with open('resource/lenses.txt') as fr:
        lenses = [instance.strip().split('\t') for instance in fr.readlines()]
        lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
        lenses_tree = create_tree(lenses, lenses_labels)
        print(lenses)
        print(lenses_labels)
        print(lenses_tree)
        create_plot(lenses_tree)
