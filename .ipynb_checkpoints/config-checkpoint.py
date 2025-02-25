# -*- coding: utf-8 -*-
# @Time : 2023/3/25 9:24
# @Author : XXX
# @Site : 
# @File : config.py
# @Software: PyCharm
import torch


class Config:
    def __init__(self, indic_kr, dm_kr, embed_dim, label):
        self.model_name = "KACNN"  # 模型名称
        self.indic_kr = indic_kr  # 指标知识表示模型
        self.dm_kr = dm_kr  # 糖尿病知识表示模型
        self.indic_embed_dim = embed_dim  # 体检指标嵌入维度
        self.dm_embed_dim = embed_dim  # 糖尿病实体嵌入维度
        self.label = label  # 标签（预测的并发症）

        self.indic_entity_path = "./benchmarks/kg_indic/entity2id.txt"  # 指标实体库
        self.dm_entity_path = f"./benchmarks/dm_redup/entity2id.txt"  # 糖尿病实体库

        self.indic_embed_path = f"./embedding_weight/kg_indic/{self.indic_kr}_{self.indic_embed_dim}.pt"  # 预训练指标实体嵌入
        self.dm_embed_path = f"./embedding_weight/kg_dm_redup/{self.dm_kr}_{self.dm_embed_dim}.pt"  # 预训练糖尿病实体嵌入

        self.train_path = f"./data/{self.label}/train.csv"  # 训练集文件路径
        self.test_path = f"./data/{self.label}/test.csv"  # 测试集文件路径

        self.model_path = f"./checkpoint/{self.model_name}_{self.indic_kr}_{self.dm_kr}_{self.indic_embed_dim}_{self.dm_embed_dim}.pth"  # 模型保存路径

        self.batch_size = 64

        self.h = 1  # 注意力的头数
        self.N = 2  # 注意力的层数

        self.dropout = 0.5  # 丢失率

        self.num_filters = 256  # 卷积核数量(channels数)
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_classes = 2  # 类别数

        self.pe_size = 256  # 体检数据编码维度
        self.gama = 0.5

        self.lr = 2e-5  # 学习率
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")