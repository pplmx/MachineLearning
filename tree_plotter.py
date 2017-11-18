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


def create_plot():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 定义一块画布,背景为白色
    figure = plt.figure(1, facecolor='white')
    # 将画布清空
    figure.clf()
    # 声明全局变量,frameon表示是否绘制坐标轴矩形
    global ax1
    ax1 = plt.subplot(111, frameon=False)
    plot_node('决策节点', (0.5, 0.1), (0.1, 0.5), decision_node)
    plot_node('叶节点', (0.8, 0.1), (0.3, 0.8), leaf_node)
    plt.show()


if __name__ == '__main__':
    create_plot()
