import random
from main import device
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import datetime
import numpy as np
from entmax import  entmax_bisect
from utils import pad_sequences, TriLinearSimilarity
import time
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class Dot_Product_Attention(nn.Module):
    """
    缩放点积注意力。
    mask为true的地方需要mask.
    返回结果和attn score
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.WQ = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.WK = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.WV = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def forward(self, query, key, value, mask=None, dropout=None):
        query = self.WQ(query)  
        key = self.WK(key)
        value = self.WV(value)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
        scores = scores.squeeze()
        if mask is not None:
            assert mask.shape == scores.shape
            scores = torch.where(mask, torch.ones_like(scores) * -1e15, scores)
        scores = scores.unsqueeze(1)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
    
    
class Cosine_Similarity_Attention(nn.Module):
    """
    余弦相似度注意力。
    mask为true的地方需要mask.
    返回结果和attn score
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.WQ = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.WK = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.WV = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def forward(self, query, key, value, mask=None, dropout=None):
        query = self.WQ(query)  
        key = self.WK(key)
        value = self.WV(value)
        scores = F.cosine_similarity(query, key, dim=-1)
        scores = scores.squeeze()
        if mask is not None:
            assert mask.shape == scores.shape
            scores = torch.where(mask, torch.ones_like(scores) * -1e15, scores)
        scores = scores.unsqueeze(1)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class AttentionUpdateGateGRUCell(nn.Module):  # Not used
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # (Wu|Wr|Wn)
        self.weight_ih = nn.Parameter(
            torch.Tensor(3 * hidden_size, input_size))
        # (Uu|Ur|Un)
        self.weight_hh = nn.Parameter(
            torch.Tensor(3 * hidden_size, hidden_size))
        if bias:
            # (b_iu|b_ir|b_in)
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            # (b_hu|b_hr|b_hn)
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / self.hidden_size ** 0.5
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x, hx, att_score=None):
        gi = F.linear(x, self.weight_ih, self.bias_ih)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)
        i_u, i_r, i_n = gi.chunk(3, 1)
        h_u, h_r, h_n = gh.chunk(3, 1)

        update_gate = torch.sigmoid(i_u + h_u)
        reset_gate = torch.sigmoid(i_r + h_r)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        if att_score is not None:
            update_gate = att_score.view(-1, 1) * update_gate
        hy = (1 - update_gate) * hx + update_gate * new_gate

        return hy


class HyperConv(nn.Module):
    def __init__(self, layers, hidden_dim=128):
        super(HyperConv, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers

    def forward(self, adjacency, embedding):
        """
        :param adjacency: 对应 D^(-1)HB^(-1)H^T。 
        :param embedding: 对应self.embedding = nn.Embedding(self.n_node, self.emb_size) node_num * ebd_size。
        """
        item_embeddings = embedding
        final = [item_embeddings]
        for i in range(self.layers):  # 多次传播，H-GNN
            item_embeddings = torch.sparse.mm(adjacency.to(device), item_embeddings)
            final.append(item_embeddings)
        final = torch.stack(final)
        item_embeddings = torch.sum(final, 0) / (self.layers + 1)  # 最终把每一层的ebd结果取平均
        item_embeddings = torch.cat((torch.zeros(self.hidden_dim).view(1, -1).to(device), item_embeddings), dim=0)
        return item_embeddings  # (node_num + 1, ebd_dim)


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        # input: [B, L, E]. 这里必须transpose(-1, -2), 因为nn.Conv1d维度必须是(B, E, L)
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        # 这里又变回了[B, L, E]
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class self_attention(nn.Module):
    def __init__(self, dim, is_dropout=False, activate='relu'):
        super().__init__()
        self.is_dropout = is_dropout
        self.dropout = nn.Dropout(0.2)
        self.dim = dim
        if activate == 'relu':
            self.activate = F.relu
        elif activate == 'selu':
            self.activate = F.selu
        self.attention_mlp = nn.Linear(dim, dim)
        
        self.self_atten_w1 = nn.Linear(dim, dim)
        self.self_atten_w2 = nn.Linear(dim, dim)
        self.LN = nn.LayerNorm(dim)
        
    def forward(self, q, k, v, mask=None, alpha_ent=2):  # 1 for softmax
        if self.is_dropout:
            q_ = self.dropout(self.activate(self.attention_mlp(q)))
        else:
            q_ = self.activate(self.attention_mlp(q))
        scores = torch.matmul(q_, k.transpose(1, 2)) / math.sqrt(self.dim)  # B, 1, d * B, d, L -> B, L
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, q.size(1), -1)
            assert mask.shape == scores.shape
            scores = torch.where(mask, torch.ones_like(scores) * -1e15, scores) 
        alpha = entmax_bisect(scores, alpha_ent, dim=-1)
        att_v = torch.matmul(alpha, v)  # B, 1, L * B, L, d
        if self.is_dropout:
            att_v = self.dropout(self.self_atten_w2(self.activate(self.self_atten_w1(att_v)))) + att_v
        else:
            att_v = self.self_atten_w2(self.activate(self.self_atten_w1(att_v))) + att_v
        att_v = self.LN(att_v)
        return att_v, alpha
    
    
class self_attention2(nn.Module):   # Not used
    def __init__(self, dim, is_dropout=False, activate='relu'):
        super().__init__()
        self.is_dropout = is_dropout
        self.dropout = nn.Dropout(0.2)
        self.dim = dim
        if activate == 'relu':
            self.activate = F.relu
        elif activate == 'selu':
            self.activate = F.selu
        self.attention_mlp = nn.Linear(dim, dim)
        
        self.self_atten_w1 = nn.Linear(dim, dim)
        self.self_atten_w2 = nn.Linear(dim, dim)
        self.LN = nn.LayerNorm(dim)
        
    def forward(self, q, k, v, mask=None, alpha_ent=2):  # 1 for softmax
        if self.is_dropout:
            q_ = self.dropout(self.activate(self.attention_mlp(q)))
        else:
            q_ = self.activate(self.attention_mlp(q))
        scores = torch.matmul(q_, k.transpose(1, 2)) / math.sqrt(self.dim)  # B, 1, d * B, d, L -> B, L
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, q.size(1), -1)
            assert mask.shape == scores.shape
            scores = torch.where(mask, torch.ones_like(scores) * -1e15, scores) 
        alpha = torch.softmax(scores, dim=-1)
        alpha_msk = torch.where(alpha > 0.6, torch.ones_like(alpha), torch.zeros_like(alpha))
        alpha = alpha * alpha_msk
        att_v = torch.matmul(alpha, v)  # B, 1, L * B, L, d
        if self.is_dropout:
            att_v = self.dropout(self.self_atten_w2(self.activate(self.self_atten_w1(att_v)))) + att_v
        else:
            att_v = self.self_atten_w2(self.activate(self.self_atten_w1(att_v))) + att_v
        att_v = self.LN(att_v)
        return att_v, alpha
    
    
class HICN(nn.Module):
    def __init__(self, dataset, item_num, max_len, hidden_dim, lr, num_heads, nei_num, nei_info,
                 batch_size, all_session, item_session_dict, sample_num, layer_num, adj, hyperedge_index,
                 user_num=None):
        super().__init__()
        self.item_num = item_num
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.nei_num = nei_num
        self.lr = lr
        self.adj = adj
        self.batch_size = batch_size
        self.layer_num = layer_num
        self.sample_num = sample_num
        self.all_session = all_session
        self.dataset = dataset
        self.item_session_dict = item_session_dict
        self.ItemEmbedding = nn.Embedding(num_embeddings=item_num + 1, embedding_dim=hidden_dim, padding_idx=0)
        if user_num is not None:
            self.UserEmbedding = nn.Embedding(num_embeddings=user_num, embedding_dim=hidden_dim)
        self.PositionEmbedding = nn.Embedding(num_embeddings=max_len, embedding_dim=hidden_dim)
        '''
        以下聚合邻居信息。
        Wo: 邻居出现次数相关
        Ws: 邻居信息聚合
        '''
        self.Wo = nn.Parameter(torch.randn(nei_num))
        self.Ws = nn.Parameter(torch.randn(nei_num, 1))
        self.item_nei, self.nei_val = nei_info
        self.item_nei = torch.from_numpy(np.array(self.item_nei)).long().to(device)
        self.nei_val = torch.from_numpy(np.array(self.nei_val)).long().to(device)
        '''
        以下初步捕捉session信息，多头注意力self-attentive
        '''
        self.W1 = nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)
        self.W2 = nn.Linear(2 * hidden_dim, num_heads, bias=False)
        self.num_heads = num_heads

        '''
        以下得到item-final-ebd
        '''
        self.MLP1 = nn.Linear(3 * hidden_dim, hidden_dim)
        self.MLP2 = nn.Linear(2 * hidden_dim, hidden_dim)
        if dataset == 'Tmall':
            self.Dropout = nn.Dropout(0.5)
        else:
            self.Dropout = nn.Dropout(0.3)
        '''
        以下得到soft-attn
        '''
        self.W3 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W4 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.q = nn.Linear(hidden_dim, 1)
        self.W5 = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.LN = nn.LayerNorm(self.hidden_dim)

        self.loss_function = torch.nn.CrossEntropyLoss()
        if self.dataset == 'Tmall':
            # AdamW -- Tmall best wd = 1e-3
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8,
                                               weight_decay=1e-3, amsgrad=False)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.1)
        else:
            # Adam
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-6)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
        self.init_parameters()
        self.HGNN = HyperConv(self.layer_num, self.hidden_dim)

        '''
        以下是cur对his的attn
        '''
        self.AUGRU = AttentionUpdateGateGRUCell(hidden_dim, hidden_dim)
        '''
        以下是邻居的attention
        '''
        self.Neighbour_Attn = Dot_Product_Attention(self.hidden_dim)
        self.Nei_Attn_Q_Ln = nn.LayerNorm(self.hidden_dim)
        self.Nei_Attn_K_Ln = nn.LayerNorm(self.hidden_dim)
        # 以下是session内的AUGRU
        self.Session_AUGRU = AttentionUpdateGateGRUCell(self.hidden_dim, self.hidden_dim)
        # 对于所有item邻居的attention
        self.lj_Attn = nn.MultiheadAttention(hidden_dim, 4)
        self.Gate = nn.Linear(2 * hidden_dim, 1, bias=False)
        self.Attn_Q_Ln = nn.LayerNorm(self.hidden_dim)
        self.Attn_K_Ln = nn.LayerNorm(self.hidden_dim)
        self.Attn_Out_Ln = nn.LayerNorm(self.hidden_dim)
        # 以下self-attention
        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(self.hidden_dim, eps=1e-8)
        # 以下融合时用的FFN
        self.FFN = PointWiseFeedForward(self.hidden_dim, 0.1)
        if self.dataset == 'Tmall':
            self.Attention_Layer_Nums = 2
        else:
            self.Attention_Layer_Nums = 2
        for _ in range(self.Attention_Layer_Nums):
            new_attn_layernorm = torch.nn.LayerNorm(self.hidden_dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(self.hidden_dim, 4, 0.1)  # num_heads, dropout_rate
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.hidden_dim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.hidden_dim, 0.1)  # dropout
            self.forward_layers.append(new_fwd_layer)
        # 以下是自注意力的entmax
        self.entmax_attention = self_attention(self.hidden_dim, True)
        self.entmax_last = self_attention(self.hidden_dim, True)
        # 衰减函数
        # self.reduce_func = 1 / torch.arange(self.max_len, 0, -1) # 1 / x
        self.reduce_func = torch.exp(-torch.arange(self.max_len - 1, -1, -1) * 0.5)
        
        self.WW0 = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.WW1 = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.BB = nn.Parameter(torch.Tensor(self.hidden_dim))
        self.FF1 = PointWiseFeedForward(self.hidden_dim, 0.1)
        self.Norm_FF1 = torch.nn.LayerNorm(self.hidden_dim, eps=1e-8)
        # 其他Session中的item
        self.Other_session_Attn_Q_Ln = nn.LayerNorm(self.hidden_dim)
        self.Other_session_Attn_K_Ln = nn.LayerNorm(self.hidden_dim)
        self.Other_session_Attn = torch.nn.MultiheadAttention(self.hidden_dim, 4, 0.1)
        self.Entmax_Other = self_attention2(self.hidden_dim, True)

    def Update_Item_Embedding(self):
        return self.HGNN(self.adj, self.ItemEmbedding.weight[1:])  # 原始HGCN

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        self.get_pad_seqs()
        self.lst_epo = 0
        self.smp = self.random_sample()
        for (name, weight) in self.named_parameters():
            if name == 'ItemEmbedding.weight':
                weight.data[1:].uniform_(-stdv, stdv)
            else:
                weight.data.uniform_(-stdv, stdv)

    def get_neighbors(self, EE=None):
        """
        I: item_num
        N: nei_num
        """
        if EE is None:
            EE = self.ItemEmbedding.weight
        Xn = EE[self.item_nei]  # I, N, d
        X1 = (torch.bmm(Xn, torch.transpose(Xn, -2, -1)) @ self.Ws).squeeze()  # I, N, N * N, 1 -> I, N, 1 -> I, N
        X2 = self.nei_val * torch.unsqueeze(self.Wo, 0)  # I, N * 1, N -> I, N
        X = X1 + X2

        X_final = torch.where(X != 0, X, torch.ones_like(X) * -1e15)
        attn = torch.softmax(X_final, dim=1).unsqueeze(1)  # I, 1, N
        return torch.bmm(attn, Xn).squeeze() # + X3 # I, d

    def get_soft_attention(self, row_session, session, mask):  # Here 'session' means Gs.
        """
        :param row_session: (None, L)
        :param session: (None, L, d)
        :return: session经过软注意力之后的结果.
        """
        session_len = torch.count_nonzero(row_session, dim=1).reshape(-1, 1)  # (None, 1)
        # 防止除0
        session_len = torch.where(session_len == 0, torch.ones_like(session_len), session_len)
        session_sum = torch.sum(session, dim=1)  # (None, d)
        session_sum = session_sum / session_len  # (None, d) # xs*
        session_sum = session_sum.unsqueeze(-2).repeat(1, self.max_len, 1)  # (None, L, d)

        x_last = self.W3(session[:, -1, :])  # None, d
        X_last = x_last.unsqueeze(1)  # None, 1, d
        X_i = self.W4(session)  # None, L, d
        # None, L, d * d, 1-> None, L, 1
        tmp = torch.relu(X_i + X_last + session_sum)
        value = self.q(tmp)
        value = torch.where(mask == 0, torch.ones_like(value) * -1e15, value)
        attn = torch.softmax(value, dim=1).transpose(-1, -2)  # None, L, 1 -> None, 1, L
        Hg = torch.bmm(attn, X_i).squeeze()  # None, d
        return self.W5(torch.cat([Hg, x_last], dim=-1))  # None, d

    def get_pad_seqs(self):
        pad_session = pad_sequences(self.all_session, self.max_len)  # None, L
        pad_session = torch.LongTensor(np.array(pad_session, dtype=np.int32))
        mask = torch.where(pad_session != 0, torch.ones_like(pad_session), torch.zeros_like(pad_session)).unsqueeze(-1)
        mask = mask.to(device)
        self.pad_session = pad_session
        self.mask = mask
        print("Paddddd Done!")

    def get_session_info(self, session, Ebd_Matrix=None):
        """
        L: max_len
        None: maybe B * max_his_len or B
        :param session: session的ebd, (None, max_len)
        :return: 多头注意力聚合后的session信息, (None, d)
        """
        if Ebd_Matrix is None:
            X = self.ItemEmbedding(session)  # None, L, d
        else:
            X = Ebd_Matrix[session.long()]  # None, L, d

        V = torch.bmm(self.W2(torch.tanh(self.W1(X))).transpose(-1, -2), X)  # None, num_heads, d
        res = torch.max(V, dim=1)[0]
        return res, V  # , Diag_SUM

    def get_last_session_info_batch(self, last_item, E):
        # print(last_item.shape)  B
        last_item = last_item.to('cpu')
        ses = self.smp[last_item.long() - 1]  # B, sample_num
        map_ses = self.pad_session[ses]  # B, sample_num, max_len
        A = self.get_session_info(map_ses.reshape(-1, self.max_len), E)[0].reshape(-1, self.sample_num,
                                                                                   self.hidden_dim)  # B, sample_num, d
        return A

    def get_session_info_batch(self, batch_item, E):   # Not used
        # batch_item: B, max_len
        ses = self.smp[batch_item.long() - 1].reshape(-1, self.sample_num)  # B * max_len, sample_num
        map_ses = self.pad_session[ses]  # B * max_len, sample_num, max_len
        # B * max_len, sample_num, d
        A = self.get_session_info(map_ses.reshape(-1, self.max_len), E)[0].reshape(-1,
                                                                                   self.sample_num,
                                                                                   self.hidden_dim)
        A = torch.sum(A, dim=1) / self.sample_num
        return A.reshape(self.batch_size, -1, self.hidden_dim)  # B, max_len, d

    def random_sample(self):
        smp = []
        # print(len(self.item_session_dict), self.item_session_dict[3], self.item_num)
        for i in range(1, self.item_num + 1):
            if i not in self.item_session_dict:
                self.item_session_dict.setdefault(i, [0])
        self.item_session_dict = dict(sorted(self.item_session_dict.items(), key=lambda x : x[0]))
        for _ in self.item_session_dict.values():
            smp.append(random.choices(_, k=self.sample_num))
        smp = torch.LongTensor(smp)
        return smp

    def get_other_session(self, session_item):   # Not used
        '''
        :param session_item: 9w * d (all_session_num, d)
        :return:
        '''
        smp = self.smp  # item_num, sample_num
        tmp = session_item[smp]  # item_num, sample_num, d
        sum = torch.sum(tmp, dim=1, keepdims=False)  # item_num, d
        sum = sum / self.sample_num
        sum = torch.cat([torch.zeros(1, self.hidden_dim).to(device), sum], dim=0)
        return sum
    
    def get_overlap(self, sessions): # sessions means session items, not used
        # 每次一个batch。sessions: B, L
        matrix = np.zeros((len(sessions), len(sessions)))
        for i in range(len(sessions)):
            seq_a = set(sessions[i])
            seq_a.discard(0)  # 移除set中的某个元素
            for j in range(i + 1, len(sessions)):  # 计算每一个session和当前session i重合了多少item
                seq_b = set(sessions[j])
                seq_b.discard(0)
                overlap = seq_a.intersection(seq_b)
                ab_set = seq_a | seq_b
                matrix[i][j] = float(len(overlap)) / float(len(ab_set))  # 对应原文的W_p, q
                matrix[j][i] = matrix[i][j]
        matrix = matrix + np.diag([1.0] * len(sessions))
        degree = np.sum(np.array(matrix), 1) # (B, )
        degree = np.diag(1.0 / degree)  # D^(-1)
        return matrix, degree
    
    def forward(self, cur, mask, id, EE, label=None):  # Here EE means the original embedding matrix.
        cur_mask = torch.where(cur != 0, torch.ones_like(cur), torch.zeros_like(cur)).squeeze().unsqueeze(-1)  # (None, L, 1)
        mask = mask.squeeze()
        positions_cur = np.tile(np.array(range(self.max_len - 1, -1, -1)), [self.batch_size, 1])  # (None, L)
        PE_cur = self.PositionEmbedding(torch.LongTensor(positions_cur).to(device))  # (None, L, d)

        nei_ebd = self.get_neighbors(EE)
        nei_ebd = torch.cat([torch.zeros(1, self.hidden_dim).to(device), nei_ebd], dim=0)
        cur = cur.reshape(-1, self.max_len)  # B, max_len
        v_s, V = self.get_session_info(cur, EE)  # None, d
        cur_info = v_s.unsqueeze(1)
        cur_info = cur_info.repeat(1, self.max_len, 1)  # None, L, d
        Es = EE[cur.long()]  # B, L, d

        seqs = EE[cur.long()]  # B, L, d
        # pos: [B, L]
        seqs += PE_cur
        seqs *= cur_mask
        Q = self.attention_layernorms[0](seqs)  # Here
        Es_prime, _ = self.entmax_attention(Q, seqs, seqs, mask, alpha_ent=2)
        cur_nei_ebd = nei_ebd[cur.long()]  # None, L, d
        
        rdd = torch.rand(self.batch_size, self.max_len)
        judge = (rdd >= self.reduce_func).to(device)
        judge |= mask
        Gs = Es + Es_prime   # None, L, d
        Gs = self.LN(Gs)
        cur_final = torch.tanh(self.MLP2(torch.cat([Gs, PE_cur], dim=-1)))  # (None, L, d)
        cur_final = cur_final * cur_mask  # (None, L, d). Here 'cur_final' is the 'Gs' is paper.

        e_s = self.get_soft_attention(cur, cur_final, cur_mask)  # None, d 即 B, d
        label_ebd = EE[cur[:, -1].long()]  # B, 1, d
        Q = self.Attn_Q_Ln(label_ebd.unsqueeze(1)).transpose(0, 1)
        K = self.Attn_K_Ln(cur_nei_ebd + PE_cur).transpose(0, 1)
        V = cur_nei_ebd.transpose(0, 1)
        tmpp = self.lj_Attn(Q, K, V, judge)
        n_s = tmpp[0].squeeze()

        lst_rh = self.get_last_session_info_batch(cur[:, -1], EE)  # B, sample_num, d
        Tri_sim = TriLinearSimilarity(self.hidden_dim, torch.nn.Sigmoid())
        sim = Tri_sim(v_s.unsqueeze(1).repeat(1, self.sample_num, 1), lst_rh)  # B, sample_num
        tar = torch.argmax(sim, dim=-1)  # B

        o_s = []
        for _ in range(self.batch_size):
            o_s.append(lst_rh[_, tar[_], :] * (sim[_, tar[_]] if sim[_, tar[_]] > 0.5 else 0))
        o_s = torch.stack(o_s, dim=0)

        beta = torch.relu(self.Gate(torch.cat([v_s, o_s], dim=-1)))

        if label is not None:
            res = e_s + v_s + n_s + 0.5 * beta * o_s   # or use MLP
        else:
            res = e_s + v_s + n_s + 0.5 * beta * o_s
        return self.Dropout(res)


def train_test(model, train_loader, test_loader, item_num, f, epoch, dataset):
    print('start training: ', datetime.datetime.now())
    total_loss = 0.0
    st = time.time()
    break_flag = False
    use_gnn = True
    if not model.lst_epo == epoch:
        model.random_sample()
        model.lst_epo = epoch
    for (ii, xx) in enumerate(train_loader):
        if break_flag:
            break
        cur, label, mask = xx
        if use_gnn:
            EE = model.Update_Item_Embedding()
        else:
            EE = model.ItemEmbedding.weight
        '''
        his: B, max_his_len, max_len
        cur: B, 1, max_len
        label: B
        '''
        cur = cur.to(device)
        label = label.to(device)
        mask = mask.to(device)
        model.zero_grad()
        final_ebd = model(cur, mask, ii, EE, label)
        if model.dataset == 'Tmall':
            item_ebd = model.ItemEmbedding.weight.detach()
        else:
            item_ebd = EE
        scores = torch.mm(final_ebd, torch.transpose(item_ebd, 1, 0))
        loss = model.loss_function(scores + 1e-8, label.long().to(device))
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if ii % 1000 == 0:
            now = time.time()
            print(f"1000-mini-batch consume: {now - st} s")
            st = now
            f.flush()
    print(f'Loss:{total_loss / len(train_loader)}')

    print("Test begins!")
    NDCG, HT, MRR = np.zeros(25), np.zeros(25), np.zeros(25)
    top_K = [5, 10, 20]
    metrics = {}
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    for (cur, label, mask) in test_loader:
        cur = cur.to(device)
        label = label.to(device)
        mask = mask.to(device)

        if model.dataset == 'Tmall':
            item_ebd = model.ItemEmbedding.weight  # (item_num + 1, D)
        else:
            item_ebd = model.Update_Item_Embedding()

        final_ebd = model(cur, mask, -1, item_ebd)  # (batch_size, D)
        scores = torch.matmul(final_ebd, torch.transpose(item_ebd[1:], -1, -2))  # (batch_size, item_num)

        if dataset in ['Tmall', 'yoo64', 'RetailRocket']:
            rk = scores.argsort().argsort()[range(scores.shape[0]), (label - 1).long().to(device)]
            rk = (model.item_num - rk - 1)
            rk = rk.to('cpu').detach().numpy()
        else:
            neg = np.random.randint(1, model.item_num, (model.batch_size, 100))
            neg = torch.from_numpy(neg).long().to(device)
            pred = torch.cat((neg, label.view(-1, 1).long().to(device)), dim=1).to(device)  # (B, 101)
            pred_ebd = model.ItemEmbedding(pred)  # -> (B, 101, D)
            scores = torch.bmm(final_ebd.unsqueeze(1), torch.transpose(pred_ebd, -2, -1))  # B, D, 101
            scores.squeeze_()  # B, 1, 101 -> B, 101

            rk = scores.argsort().argsort()[:, -1]
            rk = (100 - rk)
            rk = rk.to('cpu').detach().numpy()

        for K in top_K:
            rank = rk[rk < K]
            for rrk in rank:
                NDCG[K] += 1 / np.log2(rrk + 2)
                MRR[K] += 1 / (rrk + 1)
                HT[K] += 1

        for K in top_K:
            metrics['hit%d' % K] = HT[K] / ((len(test_loader)) * model.batch_size)
            metrics['ndcg%d' % K] = NDCG[K] / ((len(test_loader)) * model.batch_size)
            metrics['mrr%d' % K] = MRR[K] / ((len(test_loader)) * model.batch_size)

    return total_loss, metrics


def save_checkpoint(model, check_point_path, epoch):
    dirlist = os.listdir(check_point_path)
    epo = format(epoch, '04d')
    subpath = 'checkpoint_epoch_' + epo + '.pt'
    torch.save(model.state_dict(), check_point_path + '/' + subpath)


def train_epoch(num_epoch, model, train_loader, test_loader, item_num, f, dataset):
    top_K = [5, 10, 20]
    for epoch in range(num_epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        total_loss, metrics = train_test(model, train_loader, test_loader, item_num, f, epoch, dataset)
        print(metrics)
        model.scheduler.step()
        # save_checkpoint(model, './checkpoint/', epoch)
        f.flush()
