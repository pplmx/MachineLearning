#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author  : mystic
# @date    : 2018/5/17 21:22

"""
    Apriori算法
优点: 易编码实现
缺点: 在大数据集上可能较慢
适用数据类型: 数值型或标称型数据
    一般过程
1. 收集数据: 使用任意方法
2. 准备数据: 任何数据类型都可以, 因为我们只保存集合
3. 分析数据: 使用任意方法
4. 训练算法: 使用Apriori算法来找到频繁项集
5. 测试算法: 不需要测试过程
6. 使用算法: 用于法相频繁项集以及物品之间的关联规则
"""


def load_data_set():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def create_c1(data_set):
    c1 = []
    for transaction in data_set:
        for item in transaction:
            if [item] not in c1:
                c1.append([item])
    c1.sort()
    # use frozen set so we can use it as a key in a dict
    return list(map(frozenset, c1))


def scan_dict():
    pass


if __name__ == '__main__':
    print('start apriori learning')
    pass
