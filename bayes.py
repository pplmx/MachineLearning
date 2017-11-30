#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by PyCharm
# @author  : mystic
# @date    : 2017/11/21 20:40
"""
    naive bayes classifier
    优点:
        在数据较少的情况下仍然有效,可以处理多类别问题
    缺点:
        对于输入数据的准备方式较为敏感
    适用数据类型:
        标称型数据
"""
from numpy import *


def load_data_set():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vector = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return posting_list, class_vector


def create_vocabulary_list(data_set):
    # 创建一个空集
    vocabulary_set = set([])
    for document in data_set:
        # 创建两个集合的并集
        vocabulary_set = vocabulary_set | set(document)
    return list(vocabulary_set)


def set_words2vector(vocabulary_list, input_set):
    return_vector = [0]*len(vocabulary_list)
    for word in input_set:
        if word in vocabulary_list:
            return_vector[vocabulary_list.index(word)] = 1
        else:
            print('The word "%s" is not in my vocabulary!' % word)
    return return_vector


def train_naive_bayes(train_matrix, train_category):
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    probability_abusive = sum(train_category)/num_train_docs
    p0_num = zeros(num_words)
    p1_num = zeros(num_words)
    p0_denominator = 0.0
    p1_denominator = 0.0


if __name__ == '__main__':
    list_post, list_class = load_data_set()
    my_vocabulary_list = create_vocabulary_list(list_post)
    print(my_vocabulary_list)
    vec = set_words2vector(my_vocabulary_list, list_post[0])
    print(vec)
