#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by PyCharm
# @author  : mystic
# @date    : 2017/11/18 21:52
"""
    树绘制器
"""
from matplotlib import pyplot as plt

"""
    boxstyle是文本框的类型,sawtooth是锯齿形,fc是边框线的粗细
    下面的字典定义也可写作 decision_node={boxstyle:'sawtooth',fc:'0.8'} 
"""
decision_node = dict(boxstyle='sawtooth', fc='0.8')
leaf_node = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')
ax1 = None
off_x = 0.
off_y = 0.
total_width = 0.
total_depth = 0.


def get_num_leafs(my_tree):
    """
        树存储在这样的字典结构中
        {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
        no surfacing和flippers是两个分支点,前者是树根,后者是内点
    :param my_tree:
    :return:
    """
    num_leafs = 0
    # 获取树的分支点
    # first_str = my_tree.keys()[0]
    # python3:'dict_keys' object does not support indexing
    # python3中需要转成list
    first_str = list(my_tree.keys())[0]
    # 获取分支点的下层结构(子树或叶节点)
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            # 如果是dict,表示下面依然是棵子树,故需要加上子树的叶节点个数
            num_leafs += get_num_leafs(second_dict[key])
        else:
            # 如果不是,表示已至叶节点,故叶节点数+1
            num_leafs += 1
    return num_leafs


def get_tree_depth(my_tree):
    max_depth = 0
    # first_str = my_tree.keys()[0]
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + int(get_tree_depth(second_dict[key]))
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


def plot_node(node_txt, center_coordinate, parent_coordinate, node_type):
    """
        绘制节点
    :param node_txt: 节点文本信息
    :param center_coordinate: 文本框中心点,箭头所在点-坐标
    :param parent_coordinate: 射线起点-坐标
    :param node_type: 节点类型
    :return:
    """
    ax1.annotate(node_txt, xy=parent_coordinate, xycoords='axes fraction', xytext=center_coordinate,
                 textcoords='axes fraction', va='center', ha='center', bbox=node_type,
                 arrowprops=arrow_args)


def plot_mid_text(center_coordinate, parent_coordinate, txt_str):
    """
        在父子节点之间填充文本信息
    :param center_coordinate: 子节点坐标
    :param parent_coordinate: 父节点坐标
    :param txt_str: 文本信息
    :return:
    """
    mid_x = (parent_coordinate[0] + center_coordinate[0]) / 2
    mid_y = (parent_coordinate[1] + center_coordinate[1]) / 2
    ax1.text(mid_x, mid_y, txt_str)


def plot_tree(my_tree, parent_coordinate, node_txt):
    # 全局变量的引用,不需要global;修改需要
    # 特例,如果字典、列表等如果只是修改其中的元素值，可以不需要global声明
    global off_x, off_y
    num_leafs = get_num_leafs(my_tree)
    first_str = list(my_tree.keys())[0]
    # 子节点坐标
    center_coordinate = (off_x + (1.0 + num_leafs) / (2 * total_width), off_y)
    # 父子节点之间,添加文本信息
    plot_mid_text(center_coordinate, parent_coordinate, node_txt)
    # 绘制分支点(树根或内点)
    plot_node(first_str, center_coordinate, parent_coordinate, decision_node)
    second_dict = my_tree[first_str]
    off_y = off_y - 1.0 / total_depth
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            # 递归绘制子树
            plot_tree(second_dict[key], center_coordinate, str(key))
        else:
            off_x = off_x + 1.0 / total_width
            # 绘制叶节点
            plot_node(second_dict[key], (off_x, off_y), center_coordinate, leaf_node)
            plot_mid_text((off_x, off_y), center_coordinate, str(key))
    off_y = off_y + 1.0 / total_depth


def create_plot(in_tree):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 定义一块画布,背景为白色
    figure = plt.figure(1, facecolor='white')
    # 将画布清空
    figure.clf()
    axprops = dict(xticks=[], yticks=[])
    # 声明全局变量,frameon表示是否绘制坐标轴矩形
    global ax1
    ax1 = plt.subplot(111, frameon=False, **axprops)
    global total_width, total_depth, off_x, off_y
    total_width = get_num_leafs(in_tree)
    total_depth = get_tree_depth(in_tree)
    off_x = -0.5 / total_width
    off_y = 1.0
    plot_tree(in_tree, (0.5, 1.0), '')
    plt.show()


def retrieve_tree(i):
    """
        准备的一些数据
    :param i:
    :return:
    """
    list_trees = [
        {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
        {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
    ]
    return list_trees[i]


if __name__ == '__main__':
    # create_plot()
    test_tree = retrieve_tree(1)
    # print(get_num_leafs(test_tree))
    # print(get_tree_depth(test_tree))
    create_plot(test_tree)
