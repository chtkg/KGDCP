# -*- coding: utf-8 -*-
# @Time : 2023/3/25 9:25
# @Author : XXX
# @Site : 
# @File : main.py
# @Software: PyCharm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, f1_score
from tqdm import tqdm
from torch.utils.data import *
from itertools import product
from model import Model
from config import Config
from sklearn.metrics import precision_score




def load_entity_lib(lib_path):
    """加载实体库"""
    entity_lib = {}
    with open(lib_path, mode='r', encoding='utf-8') as f:
        f.readline()  # 跳过文件第一行的实体个数
        for line in f:
            entity, idx = line.split('\t')
            entity_lib[entity] = int(idx)
    return entity_lib


def entity2id(entities, entity_lib, padding):
    """将实体映射成id"""
    ids = []
    for entity in entities:
        # 超过填充边界则截断
        if len(ids) == padding:
            break
        ids.append(entity_lib["[UNK]"] if entity not in entity_lib else entity_lib[entity])
    if len(ids) < padding:
        for i in range(padding - len(ids)):
            ids.append(entity_lib["[PAD]"])
    return ids


def data_convert(data_path, indic_entity_lib, dm_entity_lib, label):
    """
    数据转换
    """
    indicator_ids = []  # 体检指标
    value_ids = []  # 体检值
    dm_ids = []  # 糖尿病实体
    labels = []  # 标签
    data = pd.read_csv(data_path);

    # 体检指标列
    cols = list(data.columns[2: 10])
    for idx in tqdm(data.index):
        # 得到input_id,
        x1 = entity2id(cols, indic_entity_lib, 8)
        x2 = entity2id(data.loc[idx, cols], indic_entity_lib, 8)
        x3 = entity2id(data.loc[idx, '口服药物'].split(' '), dm_entity_lib, 3)
        y = data.loc[idx, label]

        indicator_ids.append(x1)
        value_ids.append(x2)
        dm_ids.append(x3)
        labels.append([y])

    indicator_ids = np.array(indicator_ids, dtype=np.int64)
    value_ids = np.array(value_ids, dtype=np.int64)
    dm_ids = np.array(dm_ids, dtype=np.int64)
    labels = np.array(labels, dtype=np.int64)

    return indicator_ids, value_ids, dm_ids, labels


def load_data(indic_entity_path, dm_entity_path, train_path, test_path, label, batch_size):
    """
    加载数据
    """
    # 加载试实体库
    indic_entity_lib = load_entity_lib(config.indic_entity_path)
    dm_entity_lib = load_entity_lib(config.dm_entity_path)
    # 训练集的预处理
    train_indicator_ids, train_value_ids, train_dm_ids, train_labels = data_convert(config.train_path, indic_entity_lib,
                                                                                    dm_entity_lib, label)
    # 测试集的预处理
    test_indicator_ids, test_value_ids, tesd_dm_ids, test_labels = data_convert(config.test_path, indic_entity_lib,
                                                                                dm_entity_lib, label)
    # 包装成数据集
    train_data = TensorDataset(torch.LongTensor(train_indicator_ids),
                               torch.LongTensor(train_value_ids),
                               torch.LongTensor(train_dm_ids),
                               torch.LongTensor(train_labels))
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)  # 训练集迭代器

    test_data = TensorDataset(torch.LongTensor(test_indicator_ids),
                              torch.LongTensor(test_value_ids),
                              torch.LongTensor(tesd_dm_ids),
                              torch.LongTensor(test_labels))
    test_sampler = SequentialSampler(test_data)
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)  # 测试集迭代器

    return train_loader, test_loader


def train(model, device, train_loader, optimizer, epoch):  # 训练模型
    train_loss = 0.0
    model.train()
    best_acc = 0.0
    for batch_idx, (x1, x2, x3, y) in enumerate(train_loader):
        x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        y_pred = model(x1, x2, x3)  # 得到预测结果
        model.zero_grad()  # 梯度清零
        loss = F.cross_entropy(y_pred, y.squeeze())  # 得到loss
        train_loss += loss
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader)  # 训练集的平均损失
    return train_loss


def test(model, device, test_loader):  # 测试模型, 得到测试集评估结果
    model.eval()
    test_loss = 0.0
    all_target = []
    all_pred = []

    for batch_idx, (x1, x2, x3, y) in enumerate(test_loader):
        x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        with torch.no_grad():
            y_ = model(x1, x2, x3)
        test_loss += F.cross_entropy(y_, y.squeeze())
        pred = y_.max(-1, keepdim=True)[1]  # .max(): 2输出，分别为最大值和最大值的index
        all_target += y.view(-1).tolist()
        all_pred += pred.view(-1).tolist()

    test_loss /= len(test_loader)  # 测试集的平均损失

    acc = accuracy_score(all_target, all_pred)  # 准确率
    #pre = precision_score(all_target, all_pred, average='macro') #精确率
    pre = precision_score(all_target, all_pred, average='macro', zero_division=1)

    recall = recall_score(all_target, all_pred, average='macro')  # 召回率
   # f1 = f1_score(all_target, all_pred)
    f1 = f1_score(all_target, all_pred, average='macro')

    #     print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%, Recall: {:.2f}, F1_Score: {:.4f}'.format(
    #         test_loss,
    #         100. * acc,
    #         100. * recall,
    #         f1))
    return acc, pre, recall, f1, test_loss


def run(config):
    """
    主程序
    """
    # 加载模型
    model = Model(config)
    DEVICE = config.device
    model = model.to(DEVICE)
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    # 加载数据
    train_loader, test_loader = load_data(config.indic_entity_path, config.dm_entity_path, config.train_path,
                                          config.test_path, config.label, config.batch_size)

    train_loss_list = []
    test_loss_list = []
    acc_list = []
    pre_list = []
    recall_list = []
    f1_list = []

    best_acc = 0

    for epoch in tqdm(range(0, 1000)):  # 1000个epoch
        train_loss = train(model, DEVICE, train_loader, optimizer, epoch)
        train_loss_list.append(train_loss)
        acc, pre, recall, f1, test_loss = test(model, DEVICE, test_loader)
        # 保存精确度最高的模型
        if best_acc < acc:
            best_acc = acc
            torch.save(model.state_dict(), config.model_path)
        test_loss_list.append(test_loss)
        acc_list.append(acc)
        pre_list.append(pre)
        recall_list.append(recall)
        f1_list.append(f1)

    print(f"{config.indic_kr}_{config.dm_kr}_{config.indic_embed_dim}_{config.dm_embed_dim}_{config.label}: ")
    print("best acc is {:.4f}\n".format(max(acc_list) * 100))
    print("best pre is {:.4f}\n".format(max(pre_list) * 100))
    print("best recall is {:.4f}\n".format(max(recall_list) * 100))
    print("best f1_score is {:.4f}\n".format(max(f1_list) * 100))


if __name__ == '__main__':

   # 知识表示模型
    kr_models = ['TransE', 'TransH', 'TransR']
     #kr_models = ['TransH', 'TransR']
   # 标签
    #labels = ['脑梗', '周围神经病变']
    
    labels = ['周围神经病变']
    #embed_dim = [256, 300, 512]
    embed_dim = 512  # 只考虑维度为256的情况
   # 指标知识表示模型
    indic_kr = 'TransE'
   # 糖尿病知识表示模型
    dm_kr = 'TransE'

  

    
    for indic_kr, dm_kr, label in product(kr_models, kr_models, labels):
      # 模型参数配置
       config = Config(indic_kr, dm_kr, embed_dim, label)
       run(config)