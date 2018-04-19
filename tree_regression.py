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


def load_data_set(filename):  # general function to parse tab -delimited floats
    data_list = []  # assume last column is target value
    with open(filename) as fr:
        for line in fr.readlines():
            cur_line = line.strip().split('\t')
            flt_line = map(float, cur_line)  # map all elements to float()
            data_list.append(flt_line)
        return data_list

