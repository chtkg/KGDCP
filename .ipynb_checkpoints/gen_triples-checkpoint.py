# -*- coding: utf-8 -*-
# @Time : 2023/3/24 10:29
# @Author : XXX
# @Site : 
# @File : triples.py
# @Software: PyCharm
# @Desc : 依据指标参考范围生成三元组
import os

# 实体、关系、三元组保存地址
marks_path = "./benchmarks/kg_indic/"

# 体检指标参考范围
indicator_dict = {
    "BMI": [100, 185, 240, 400],  # 无单位
    "糖化血红蛋白": [0, 427, 607, 2000],  # %(0.01)
    "肌酐": [0, 57, 111, 1200],  # umol/L()
    "甘油三酯": [0, 45, 181, 1600],  # mmol/L(0.01)
    "胆固醇": [0, 286, 610, 1000],  # mmol/L(0.01)
    "低密度脂蛋白": [0, 0, 337, 1000],  # mmol/L(0.01)
    "高密度脂蛋白": [0, 116, 142, 1000],  # mmol/L(0.01)
    "脂蛋白a": [0, 0, 300, 1000]  # mg/L(1)
}

# 测量值实体
bmi = ['{:.1f}'.format(i*0.1) for i in range(0,401)]
per = ["{:.2f}%".format(i * 0.01) for i in range(0,2001)]
umol_l = ["{}umol/L".format(i) for i in range(0,1201)]
mmol_l = ["{:.2f}mmol/L".format(i*0.01) for i in range(0,1601)]
mg_l = ["{}mg/L".format(i) for i in range(0,1001)]
test_values = bmi + per + umol_l + mmol_l + mg_l

# 体检指标对应单位
indicator_unit = {
    "BMI": bmi,
    "糖化血红蛋白": per,
    "肌酐": umol_l,
    "甘油三酯": mmol_l,
    "胆固醇": mmol_l,
    "低密度脂蛋白": mmol_l,
    "高密度脂蛋白": mmol_l,
    "脂蛋白a": mg_l
}

# 将实体写入文件
entity_dict = {}  # 实体字典
entity_id = 1  # 实体id
with open(marks_path + "entity2id.txt", mode="w", encoding="utf-8") as entity_file:
    # 未知实体
    entity_file.write("[UNK]" + "\t" + str(0) + "\n")
    entity_dict["UNK"] = 0
    # 体检指标实体
    for indicator in indicator_dict.keys():
        entity_file.write(indicator + "\t" + str(entity_id) + "\n")
        entity_dict[indicator] = entity_id
        entity_id += 1
    # 检测值实体
    for val in test_values:
        entity_file.write(val + "\t" + str(entity_id) + "\n")
        entity_dict[val] = entity_id
        entity_id += 1

# 将关系实体写入文件
relations = ["严重偏低", "中等偏低", "轻微偏低", "正常", "轻微偏高", "中等偏高", "严重偏高"]  # 关系
rel_id = 0  # 实体id
# 将关系写入文件
with open(marks_path + "relation2id.txt", mode="w", encoding="utf-8") as relation_file:
    for relation in relations:
        relation_file.write(relation + "\t" + str(rel_id) + "\n")
        rel_id += 1


def divide_scope(scope, unit):
    """
    划分体检指标区间
    :param scope: 体检指标参考范围, [最低值，正常范围，最大值]
    :param unit: 检测值单位
    :return:
    """
    scope_list = []

    low = scope[1] - scope[0]
    high = scope[3] - scope[2]

    scope_list.append(unit[int(scope[0]): int(scope[0] + low * 0.3)])
    scope_list.append(unit[int(scope[0] + low * 0.3): int(scope[0] + low * 0.6)])
    scope_list.append(unit[int(scope[0] + low * 0.6): int(scope[1])])
    scope_list.append(unit[int(scope[1]): int(scope[2] + 1)])
    scope_list.append(unit[int(scope[2] + 1): int(scope[2] + high * 0.4)])
    scope_list.append(unit[int(scope[2] + high * 0.4): int(scope[2] + high * 0.7)])
    scope_list.append(unit[int(scope[2] + high * 0.7): int(scope[3] + 1)])
    return scope_list


# 划分每个体检指标检测值区间
triple_dict = {}
for indic, unit in indicator_unit.items():
    triple_dict[indic] = divide_scope(indicator_dict[indic], unit)

# 生成三元组并写入文件
with open(marks_path + "triple2id.txt", mode="w", encoding="utf-8") as triple_file:
    for key, value in triple_dict.items():
        r = 0
        for v in value:
            if v is not None:
                for i in v:
                    triple_file.write(str(entity_dict[key]) + " " + str(entity_dict[i]) + " " + str(r) + "\n")
            r += 1

# 按 6: 2: 2 划分训练集、测试集、验证集
train_file = open(marks_path + "train2id.txt", mode="w", encoding="utf-8")
test_file = open(marks_path + "test2id.txt", mode="w", encoding="utf-8")
valid_file = open(marks_path + "valid2id.txt", mode="w", encoding="utf-8")
with open(marks_path + "triple2id.txt", mode="r", encoding="utf-8") as f:
    i = 0
    f.readline()
    for line in f:
        if i % 5 in (0, 2, 4):
            train_file.writelines(line)
        elif i % 5 == 1:
            test_file.writelines(line)
        else:
            valid_file.writelines(line)
        i += 1

train_file.close()
test_file.close()
valid_file.close()


# 统计文件行数
def add_line_number(file_path):
    i = 0
    with open(file_path, mode='r', encoding='utf-8') as f:
        for line in f:
            i += 1
    with open(file_path, mode='r+', encoding='utf-8') as f:
        old = f.read()
        f.seek(0)
        f.write(str(i) + "\n")
        f.write(old)


add_line_number(marks_path + "entity2id.txt")
add_line_number(marks_path + "relation2id.txt")
add_line_number(marks_path + "triple2id.txt")
add_line_number(marks_path + "train2id.txt")
add_line_number(marks_path + "test2id.txt")
add_line_number(marks_path + "valid2id.txt")
