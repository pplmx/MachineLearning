#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by PyCharm
# @author  : mystic
# @date    : 2018/3/20 20:39

"""
    SVM: Support Vector Machine - 支持向量机
    优点:
        泛化错误率低,计算开销不大,结果易解释
    缺点:
        对参数调节和核函数的选择敏感,原始分类器不加修改仅适用于处理二类问题
    适用数据范围:
        数值型和标称型数据
    SVM的一般流程
        [1]收集数据: 可以使用任意方法
        [2]准备数据: 需要数值型数据
        [3]分析数据: 有助于可视化分隔超平面
        [4]训练算法: SVM的大部分时间都源自训练,该训练主要实现两个参数的调优
        [5]测试算法: 十分简单的计算过程就可以实现
        [6]使用算法: 几乎所有分类问题都可以使用SVM,值得一提的是,SVM本身是一个二类分类器
                    对多类问题应用,SVM需要对代码做一些修改
"""
import random


def load_data_set(file_name):
    data_mat = []
    label_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = line.strip().split('\t')
        data_mat.append([float(line_arr[0]), float(line_arr[1])])
        label_mat.append(float(line_arr[2]))
    return data_mat, label_mat


def select_jrand(i, m):
    j = i
    while j==i:
        j = int(random.uniform(0, m))
    return j


def clip_alpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


