#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by PyCharm
# @author  : mystic
# @date    : 2017/11/13 20:39
"""
    k-NearestNeighbor
    核心思想:
        如果一个样本在特征空间中的k个最相邻的样本中的大多数属于某一个类别，
        则该样本也属于这个类别，并具有这个类别上样本的特性
    优点:
        精度高、对异常值不敏感、无数据输入假定
    缺点：
        计算复杂度高、空间复杂度高
    适用数据范围：
        数值型和标称型
"""