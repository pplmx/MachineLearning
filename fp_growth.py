#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author  : mystic
# @date    : 2018/5/23 20:35
"""
    FP_growth算法(Frequent Pattern)
优点: 一般要快于Apriori
缺点: 实现比较困难,在某些数据集上性能会下降
适用数据类型: 标称型数据
    一般流程
1.收集数据: 使用任意方法
2.准备数据: 由于存储的是集合,所以需要离散数据(如果要处理连续数据,需要将它们良华为离散值)
3.分析数据: 使用任意方法
4.训练算法: 构建一个FP树,并对树进行挖掘
5.测试算法: 没有测试过程
6.使用算法: 可用于识别经常出现的元素项,从而用于制定决策、推荐元素或进行预测等
"""


class Tree:
    def __init__(self, tree_name, occur_times, parent_node):
        self.name = tree_name
        self.count = occur_times
        self.nodeLink = None
        self.parent = parent_node  # needs to be updated
        self.children = {}

    def increase(self, occur_times):
        self.count += occur_times

    def display(self, idx=1):
        print('  ' * idx, self.name, ' ', self.count)
        for child in self.children.values():
            child.display(idx + 1)


def create_tree(data_set, min_support=1):
    """
    create FP-tree from data_set but don't mine
    :param data_set:
    :param min_support:
    :return:
    """
    header_table = {}
    # go over dataSet twice
    for transaction in data_set:
        for item in transaction:
            header_table[item] = header_table.get(item, 0) + data_set[transaction]
