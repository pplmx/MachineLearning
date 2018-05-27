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
    # go over data_set twice
    for transaction in data_set:
        for item in transaction:
            header_table[item] = header_table.get(item, 0) + data_set[transaction]

    for k in list(header_table.keys()):  # remove items not meeting min_support
        if header_table[k] < min_support:
            del (header_table[k])
    frequent_item_set = set(header_table.keys())
    if len(frequent_item_set) == 0:
        return None, None  # if no items meet min support -->get out
    for k in header_table:
        header_table[k] = [header_table[k], None]  # reformat header_table to use Node link 
    ret_tree = Tree('Null Set', 1, None)  # create tree
    for transaction_set, count in data_set.items():  # go through data_set 2nd time
        local_dict = {}
        for item in transaction_set:  # put transaction items in order
            if item in frequent_item_set:
                local_dict[item] = header_table[item][0]
        if len(local_dict) > 0:
            ordered_items = [v[0] for v in sorted(local_dict.items(), key=lambda p: p[1], reverse=True)]
            update_tree(ordered_items, ret_tree, header_table, count)  # populate tree with ordered frequent item set
    return ret_tree, header_table  # return tree and header table


def update_tree(items, input_tree, header_table, count):
    if items[0] in input_tree.children:  # check if orderedItems[0] in retTree.children
        input_tree.children[items[0]].increase(count)  # increment count
    else:  # add items[0] to inTree.children
        input_tree.children[items[0]] = Tree(items[0], count, input_tree)
        if header_table[items[0]][1] is None:  # update header table
            header_table[items[0]][1] = input_tree.children[items[0]]
        else:
            update_header(header_table[items[0]][1], input_tree.children[items[0]])
    if len(items) > 1:  # call updateTree() with remaining ordered items
        update_tree(items[1::], input_tree.children[items[0]], header_table, count)


def update_header(node2test, target_node):  # this version does not use recursion
    while node2test.nodeLink is not None:  # Do not use recursion to traverse a linked list!
        node2test = node2test.nodeLink
    node2test.nodeLink = target_node


def ascend_tree(leaf_node, prefix_path):  # ascends from leaf node to root
    if leaf_node.parent is not None:
        prefix_path.append(leaf_node.name)
        ascend_tree(leaf_node.parent, prefix_path)


def find_prefix_path(tree_node):  # treeNode comes from header table
    condition_pattern = {}
    while tree_node is not None:
        prefix_path = []
        ascend_tree(tree_node, prefix_path)
        if len(prefix_path) > 1:
            condition_pattern[frozenset(prefix_path[1:])] = tree_node.count
        tree_node = tree_node.nodeLink
    return condition_pattern


def mine_tree(header_table, min_support, prefix, frequent_item_list):
    big_l = [v[0] for v in sorted(header_table.items(), key=lambda p: p[1])]  # (sort header table)
    for basePat in big_l:  # start from bottom of header table
        new_frequent_set = prefix.copy()
        new_frequent_set.add(basePat)
        frequent_item_list.append(new_frequent_set)
        condition_pattern_bases = find_prefix_path(header_table[basePat][1])
        # 2. construct condition FP-tree from condition pattern base
        my_condition_tree, my_head = create_tree(condition_pattern_bases, min_support)
        if my_head is not None:  # 3. mine condition FP-tree
            mine_tree(my_head, min_support, new_frequent_set, frequent_item_list)


def load_simple_data():
    simple_data = [['r', 'z', 'h', 'j', 'p'],
                   ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
                   ['z'],
                   ['r', 'x', 'n', 'o', 's'],
                   ['y', 'r', 'x', 'z', 'q', 't', 'p'],
                   ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simple_data


def create_initial_set(data_set):
    ret_dict = {}
    for trans in data_set:
        ret_dict[frozenset(trans)] = 1
    return ret_dict


if __name__ == '__main__':
    with open('resource/kosarak.dat') as fr:
        parsed_data_ = [line.split() for line in fr.readlines()]
        initial_set_ = create_initial_set(parsed_data_)
        fp_tree_, header_tab_ = create_tree(initial_set_, 100000)
        frequent_list_ = []
        mine_tree(header_tab_, 100000, set([]), frequent_list_)
        print(len(frequent_list_))
        print(frequent_list_)
