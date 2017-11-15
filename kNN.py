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

from numpy import *
import operator
import matplotlib.pyplot as plt
from os import listdir


def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 1], [0, 0.1]])
    labels = ['A', 'C', 'B', 'D']
    return group, labels


def classify(in_x, data_set, labels, k):
    """
    分类器
    :param in_x: 用于分类的输入向量
    :param data_set: 输入的训练样本集
    :param labels: 标签向量
    :param k: 用于选择最近邻居的数目
    :return:
    """
    # 获取data_set的第一维长度
    data_set_size = data_set.shape[0]
    # 分别计算输入向量与data_set集合中各点的向量差,并存入数组中
    diff_arr = tile(in_x, (data_set_size, 1)) - data_set
    # 平方
    sq_diff_arr = diff_arr ** 2
    # 求平方和
    sq_distinces = sq_diff_arr.sum(axis=1)
    # 开根,得各点与输入向量的距离值集合
    distinces = sq_distinces ** 0.5
    # 排序,升序(返回结果为索引,如[17,23,1,0],排序后返回[3,2,0,1])
    sorted_dist_indices = distinces.argsort()
    # print('最近的点:%s' % labels[sorted_dist_indices[0]])
    # 存储最近的k个点
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_dist_indices[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    # print(class_count)
    # 根据字典class_count的value进行降序排列
    # 在最近点案例中,value都是1,下面的排序等于没做
    sorted_class_count = sorted(class_count.items(),
                                key=operator.itemgetter(1), reverse=True)
    # print(sorted_class_count)
    return sorted_class_count[0][0]


def file2array(filename):
    # 获取文件行数
    fr = open(filename)
    array_lines = fr.readlines()
    amount = len(array_lines)
    # 创建第一维长度为amount,第二维长度为3的零填充数组
    # 因为我们需要存储的数据第二维长度为3,故我们设置固定长度3
    return_array = zeros((amount, 3))
    # 创建空list
    class_label_vector = []
    index = 0
    for line in array_lines:
        # strip([chars]) 去除头尾的字符,默认去除空格字符
        line = line.strip()
        list_from_line = line.split('\t')
        # 获取前三个元素,赋值给return_array(这里采用了多维切片)
        return_array[index, :] = list_from_line[0:3]
        # 负索引从后获取,-1获取最后一个元素
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    # 返回tuple
    return return_array, class_label_vector


def show_data_in_chart():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    dating_data_arr, dating_labels = file2array('resource/datingTestSet2.txt')
    # new figure
    fig = plt.figure()
    # 在1行1列,第一个子图作画;如233,在2行3列共有6个子图的图中,在第3个子图中作画
    ax = fig.add_subplot(111)
    # 设置标题
    ax.set_title('Appointment Data Analysis')
    # Helen提供的数据,三列分别是:每年获得的飞行常客里程数,玩视频游戏所耗时间百分比,每周消费的冰淇淋公升数
    # 1:not like    2:general like      3:very like
    # 因为我们最后显示的是第一列和第二列数据,故如下设置备注信息
    plt.xlabel('每年获得的飞行常客里程数')
    plt.ylabel('玩视频游戏所耗时间百分比')
    # ax.scatter(dating_data_arr[:, 1], dating_data_arr[:, 2])
    # ax.scatter(dating_data_arr[:, 1], dating_data_arr[:, 2], 15.0*array(dating_labels), 15.0*array(dating_labels))
    # 第一列与第二列的数据,显示效果更优
    # ax.scatter(dating_data_arr[:, 0], dating_data_arr[:, 1])
    # 这种方式没有图例,难以区分
    # ax.scatter(dating_data_arr[:, 0], dating_data_arr[:, 1], 15.0 * array(dating_labels), 15.0 * array(dating_labels))
    # 添加图例
    length = dating_data_arr.shape[0]
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    x3 = []
    y3 = []
    for i in range(length):
        if dating_labels[i] == 1:
            x1.append(dating_data_arr[i, 0])
            y1.append(dating_data_arr[i, 1])
        elif dating_labels[i] == 2:
            x2.append(dating_data_arr[i, 0])
            y2.append(dating_data_arr[i, 1])
        else:
            x3.append(dating_data_arr[i, 0])
            y3.append(dating_data_arr[i, 1])
    type1 = ax.scatter(x1, y1, c='red')
    type2 = ax.scatter(x2, y2, c='green')
    type3 = ax.scatter(x3, y3, c='blue')
    ax.legend([type1, type2, type3], ["not like", "general like", "very like"], loc=2)
    plt.show()


def auto_norm(data_set):
    """
    归一化特征值:自动将数据集转化为0到1区间内的值
    由于里程数远远大于其他特征值,对结果影响过大
    而Helen认为三者同等重要,故采用归一化处理
    :param data_set:
    :return:
    """
    # 取第一维度的最小值
    """
        >>> sh = array([
                [[1, 1],[8, 18],[100, 3],[2, 4]],
                [[1, 90],[21, 2],[11, 3],[19, 4]]
            ])
        >>> shape(sh)
        (2,4,2)
        >>> sh.max()
        100
        >>> sh.min()
        1
        >>> sh.max(0)
        array([[  1,  90],
               [ 21,  18],
               [100,   3],
               [ 19,   4]])
        >>> sh.min(0)
        array([[ 1,  1],
               [ 8,  2],
               [11,  3],
               [ 2,  4]])
    """
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals
    # 创建一个与data_set各维长度均相同的零填充数组
    # norm_data_set = zeros(shape(data_set))
    length = data_set.shape[0]
    norm_data_set = data_set - tile(min_vals, (length, 1))
    norm_data_set = norm_data_set / tile(ranges, (length, 1))
    return norm_data_set, ranges, min_vals


def dating_class_test():
    """
    测试代码
    :return:
    """
    # 用于测试的数据,占总数据的百分比
    # [如:已有100条数据,其中90条作为样本训练数据,剩余10条用于测试算法,检测算法的正确率]
    test_ratio = 0.10
    dating_data_arr, dating_labels = file2array('resource/datingTestSet2.txt')
    norm_arr, ranges, min_vals = auto_norm(dating_data_arr)
    length = norm_arr.shape[0]
    # 测试数据总数
    num_test_data = int(length * test_ratio)
    error_count = 0.0
    for i in range(num_test_data):
        classifier_result = classify(norm_arr[i, :], norm_arr[num_test_data:length, :],
                                     dating_labels[num_test_data:length], 3)
        print('The classifier came back with: %d, the real answer is: %d'
              % (classifier_result, dating_labels[i]))
        # 如果分类器结果和真实数据,不同;error_count+1
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    print('The total error rate is: %f' % (error_count / num_test_data))


def classify_person():
    """
    预测函数
    :return:
    """
    result_list = ['not like', 'general like', 'very like']
    game_percent = float(input('percentage of time spent playing video games?'))
    fly_miles = float(input('frequent flier miles earned per year?'))
    how_much_ice_cream = float(input('liters of ice cream consumed per week?'))
    dating_data_arr, dating_labels = file2array('resource/datingTestSet2.txt')
    norm_arr, ranges, min_vals = auto_norm(dating_data_arr)
    in_arr = array([fly_miles, game_percent, how_much_ice_cream])
    classifier_result = classify((in_arr - min_vals) / ranges, norm_arr, dating_labels, 3)
    print('You will probably like this person: %s' % result_list[classifier_result - 1])


def img2vector(filename):
    """
    将32*32的二进制图像矩阵转化为1*1024的向量
    :param filename:
    :return:
    """
    return_vector = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            return_vector[0, 32*i+j] = int(line[j])
    return return_vector


def handwriting_class_test():
    handwriting_labels = []
    # 训练数据
    training_file_list = listdir('resource/digits/trainingDigits')
    length = len(training_file_list)
    training_arr = zeros((length, 1024))
    for i in range(length):
        # 获取文件名,含后缀
        filename_str = training_file_list[i]
        file_str = filename_str.split('.')[0]
        # 获取文件中存储二进制图像,表示的数字
        class_num_str = int(file_str.split('_')[0])
        handwriting_labels.append(class_num_str)
        # 将各文件生成的1*1024向量分别存入training_arr
        training_arr[i, :] = img2vector('resource/digits/trainingDigits/%s' % filename_str)
    # 测试数据
    test_file_list = listdir('resource/digits/testDigits')
    error_count = 0.0
    length = len(test_file_list)
    for i in range(length):
        filename_str = test_file_list[i]
        file_str = filename_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        # 读取一个测试文件,并生成1*1024的向量,赋值给vector_under_test
        vector_under_test = img2vector('resource/digits/testDigits/%s' % filename_str)
        classifier_result = classify(vector_under_test, training_arr, handwriting_labels, 3)
        print('The classifier came back with: %d,the real answer is: %d'
              % (classifier_result, class_num_str))
        if classifier_result != class_num_str:
            error_count += 1.0
    print('The total number of errors is: %d' % error_count)
    print('The total error rate is: %f' % (error_count/length))


if __name__ == '__main__':
    # show_data_in_chart()
    # 通过调整test_ratio的比例,以及k的值,使结果最优
    # dating_class_test()
    # classify_person()
    # test_vector = img2vector('resource/digits/testDigits/0_0.txt')
    # print(test_vector[0, 0:32])
    handwriting_class_test()
