#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author  : mystic
# @date    : 2018/5/6 17:11

"""
        K-均值聚类的一般流程
1. 收集数据: 使用任意方法
2. 准备数据: 需要数值型数据来计算距离, 也可以将标称型数据映射为二值型数据, 再用于计算
3. 分析数据: 使用任意方法
4. 训练算法: 不适用于无监督学习, 即无监督学习没有训练过程
5. 测试算法: 应用聚类算法, 观察结果
            可以使用量化的误差指标,如误差平方和来评价算法的结果
6. 使用算法: 可以用于所希望的任何应用
            通常情况下, 簇质心可以代表整个簇的数据来作出决策
"""


def load_data_set(filename):  # general function to parse tab -delimited floats
    data_list = []  # assume last column is target value
    with open(filename) as fr:
        for line in fr.readlines():
            cur_line = line.strip().split('\t')
            # TODO the return type has changed in python3
            # before change: map(float, cur_line)
            # after change: list(map(float, cur_line))
            flt_line = list(map(float, cur_line))  # map all elements to float()
            data_list.append(flt_line)
        return data_list
