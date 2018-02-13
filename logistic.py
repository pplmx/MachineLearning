#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by PyCharm
# @author  : mystic
# @date    : 2/13/2018 16:38

"""
Logistic回归的一般过程
    (1)收集数据: 采用任意方法收集数据
    (2)准备数据: 由于需要进行距离计算,因此需求数据类型为数值型.另外,结构化数据格式则最佳
    (3)分析数据: 采用任意方法对数据进行分析
    (4)训练算法: 大部分时间将用于训练,训练的目的是为了找到最佳的分类回归系数
    (5)测试算法: 一旦训练步骤完成,分类将会很快
    (6)使用算法:
                首先,我们需要输入一些数据,并将其转化成对应的结构化数据;
                接着,基于训练好的回归系数就可以对这些数值进行简单的回归计算,判定它们属于哪个类别;
                在这之后,我么就可以在输出的类别上做一些其他分析工作
"""