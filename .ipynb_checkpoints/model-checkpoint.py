import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable


#  体检指标嵌入层
class IndicEmbeddings(nn.Module):
    def __init__(self, embed_dim, pretrained_embedding=None):
        super(IndicEmbeddings, self).__init__()
        if pretrained_embedding == None:
            self.lut = nn.Embedding(6214, embed_dim)
        else:
            weight = torch.load(pretrained_embedding)
            weight.require_grad = False
            self.lut = nn.Embedding.from_pretrained(weight)
        self.embed_dim = embed_dim

    def forward(self, x1, x2):
        embeding1 = self.lut(x1) * math.sqrt(self.embed_dim)
        embeding2 = self.lut(x2) * math.sqrt(self.embed_dim)
        return embeding2 - embeding1

    

# 糖尿病实体实体嵌入层
class DmEmbeddings(nn.Module):
    def __init__(self, embed_dim, pretrained_embedding=None):
        super(DmEmbeddings, self).__init__()
        if pretrained_embedding == None:
            self.lut = nn.Embedding(4634, embed_dim)
        else:
            weight = torch.load(pretrained_embedding)
            weight.require_grad = False
            self.lut = nn.Embedding.from_pretrained(weight)
        self.embed_dim = embed_dim

    def forward(self, x):
        embeding = self.lut(x) * math.sqrt(self.embed_dim)
        return embeding


# 自注意力机制
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class Token2TokenAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(Token2TokenAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value,
                                 dropout=self.dropout)
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


# 体检指标编码
class IndicCoding(nn.Module):
    def __init__(self, embed_dim, embed_path, num_filters, filter_sizes, pe_size):
        """
         embed_dim：嵌入维度
         embed_path：与训练嵌入地址
         num_filters：过滤器个数
         filter_sizes：过滤器大小 
         pe_size：体检向量维度
        """
        super(IndicCoding, self).__init__()
        # 嵌入层
        self.embed = IndicEmbeddings(embed_dim, embed_path)
        # 注意力层
        self.attn = Token2TokenAttention(1, embed_dim)
        # 卷积层
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, embed_dim)) for k in filter_sizes])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(filter_sizes), pe_size)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, indic_id, val_id):
        x = self.embed(indic_id, val_id)
        out = self.attn(x, x, x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


#  诊疗信息编码
class DiagCoding(nn.Module):
    def __init__(self, embed_dim, embed_path, pe_size, gama=0.5):
        """
         embed_dim：嵌入维度
         pe_size：体检数据编码维度 
         gama：不同注意力权重
        """
        super(DiagCoding, self).__init__()

        self.gama = gama
        da = pe_size
        db = int(da / 2)

        self.embed = DmEmbeddings(embed_dim, embed_path)  # 嵌入层
        self.W1 = nn.Linear(pe_size + embed_dim, da)
        self.w1 = nn.Linear(da, 1, bias=False)
        self.W2 = nn.Linear(embed_dim, db)
        self.w2 = nn.Linear(db, 1, bias=False)

    def e2p_attention(self, c, pe):
        # entity2PE 注意力
        # c: batch_size, seq_len, embed_dim
        # p: batch_size, pe_size
        pe = pe.unsqueeze(1)
        pe = pe.expand(pe.size(0), c.size(1), pe.size(2))
        c_pe = torch.cat((c, pe), -1)  # batch_size, seq_len, embed_dim+pe_size
        c_pe = self.w1(torch.tanh(self.W1(c_pe)))  # batch_size, seq_len, 1
        alpha = F.softmax(c_pe.squeeze(-1), -1)  # batch_size, seq_len

        return alpha

    def s2t_attention(self, c):
        # source2token 注意力
        # c: batch_size, concept_seq_len, embedding_dim
        c = self.w2(torch.tanh(self.W2(c)))  # batch_size, concept_seq_len, 1
        beta = F.softmax(c.squeeze(-1), -1)  # batch_size, concept_seq_len

        return beta

    def forward(self, dm_entity_id, pe):
        embedding = self.embed(dm_entity_id)  # input_: batch_size, concept_seq_len, emb_dim
        alpha = self.e2p_attention(embedding, pe)  # batch_size, concept_seq_len
        beta = self.s2t_attention(embedding)  # batch_size, concept_seq_len
        atten = F.softmax(self.gama * alpha + (1 - self.gama) * beta, -1)  # batch_size, concept_seq_len
        pc = torch.bmm(atten.unsqueeze(1), embedding).squeeze(1)  # batch_size, emb_dim

        return pc


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # 体检指标编码
        self.indic_coding = IndicCoding(config.indic_embed_dim, config.indic_embed_path, config.num_filters,
                                        config.filter_sizes, config.pe_size)
        # 诊疗信息编码
        self.diag_coding = DiagCoding(config.dm_embed_dim, config.dm_embed_path, config.pe_size, config.gama)
        # 全连接层
        self.fc = nn.Linear(config.pe_size + config.dm_embed_dim, config.num_classes)

    def forward(self, indic_id, val_id, dm_entity_id):
        pe = self.indic_coding(indic_id, val_id)
        pc = self.diag_coding(dm_entity_id, pe)
        p = torch.cat((pe, pc), -1)
        out = self.fc(p)
        return out
