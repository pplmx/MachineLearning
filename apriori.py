#!/usr/bin/env python
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


def scan_dict(d, c_k, min_support):
    ss_cnt = {}
    for tid in d:
        for can in c_k:
            if can.issubset(tid):
                if can not in ss_cnt:
                    ss_cnt[can] = 1
                else:
                    ss_cnt[can] += 1
    num_items = float(len(d))
    ret_list = []
    support_data = {}
    for key in ss_cnt:
        support = ss_cnt[key] / num_items
        if support >= min_support:
            ret_list.insert(0, key)
        support_data[key] = support
    return ret_list, support_data


def apriori_gen(l_k, k):  # creates c_k
    ret_list = []
    len_lk = len(l_k)
    for i in range(len_lk):
        for j in range(i + 1, len_lk):
            l1 = list(l_k[i])[: k - 2]
            l2 = list(l_k[j])[: k - 2]
            l1.sort()
            l2.sort()
            if l1 == l2:  # if first k-2 elements are equal
                ret_list.append(l_k[i] | l_k[j])  # set union
    return ret_list


def apriori(data_set, min_support=0.5):
    c1 = create_c1(data_set)
    d = list(map(set, data_set))
    l1, support_data = scan_dict(d, c1, min_support)
    l1_to_list = [l1]
    k = 2
    while len(l1_to_list[k - 2]) > 0:
        c_k = apriori_gen(l1_to_list[k - 2], k)
        l_k, sup_k = scan_dict(d, c_k, min_support)  # scan DB to get Lk
        support_data.update(sup_k)
        l1_to_list.append(l_k)
        k += 1
    return l1_to_list, support_data


def generate_rules(
    l, support_data, min_confidence=0.7
):  # supportData is a dict coming from scanD
    big_rule_list = []
    for i in range(1, len(l)):  # only get the sets with two or more items
        for freqSet in l[i]:
            h1 = [frozenset([item]) for item in freqSet]
            if i > 1:
                rules_from_consequence(
                    freqSet, h1, support_data, big_rule_list, min_confidence
                )
            else:
                calc_confidence(
                    freqSet, h1, support_data, big_rule_list, min_confidence
                )
    return big_rule_list


def calc_confidence(frequent_set, H, support_data, brl, min_confidence=0.7):
    pruned_h = []  # create new list to return
    for consequence in H:
        conf = (
            support_data[frequent_set] / support_data[frequent_set - consequence]
        )  # calc confidence
        if conf >= min_confidence:
            print(frequent_set - consequence, "-->", consequence, "conf:", conf)
            brl.append((frequent_set - consequence, consequence, conf))
            pruned_h.append(consequence)
    return pruned_h


def rules_from_consequence(frequent_set, h, support_data, brl, min_confidence=0.7):
    m = len(h[0])
    if len(frequent_set) > (m + 1):  # try further merging
        hmp1 = apriori_gen(h, m + 1)  # create Hm+1 new candidates
        hmp1 = calc_confidence(frequent_set, hmp1, support_data, brl, min_confidence)
        if len(hmp1) > 1:  # need at least two sets to merge
            rules_from_consequence(
                frequent_set, hmp1, support_data, brl, min_confidence
            )


def print_rules(rule_list, item_meaning):
    for ruleTup in rule_list:
        for item in ruleTup[0]:
            print(item_meaning[item])
        print("           -------->")
        for item in ruleTup[1]:
            print(item_meaning[item])
        print("confidence: %f\n" % ruleTup[2])


if __name__ == "__main__":
    print("start apriori learning")
    pass
