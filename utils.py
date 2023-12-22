# -*- coding = utf-8 -*-
# @Time : 2022-11-20 16:12
# @Author: gla1ve
# @File: LeGao_utils.py.py
# @Software: PyCharm
# 人生苦短， 我用python(划掉) Java
import numpy as np
import torch
from torch import nn
import math
from torch.utils.data import DataLoader, Dataset
from scipy.sparse import csr_matrix
import sys
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DotProductSimilarity(nn.Module):
 
    def __init__(self, scale_output=False):
        super(DotProductSimilarity, self).__init__()
        self.scale_output = scale_output
 
    def forward(self, tensor_1, tensor_2):
        result = (tensor_1 * tensor_2).sum(dim=-1)
        if self.scale_output:
            result /= math.sqrt(tensor_1.size(-1))
        return result
    
    
class BiLinearSimilarity(nn.Module):
 
    def __init__(self, tensor_1_dim, tensor_2_dim, activation=None):
        super(BiLinearSimilarity, self).__init__()
        self.weight_matrix = nn.Parameter(torch.Tensor(tensor_1_dim, tensor_2_dim)).to(device)
        self.bias = nn.Parameter(torch.Tensor(1)).to(device)
        self.activation = activation
        self.reset_parameters()
 
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_matrix)
        self.bias.data.fill_(0)
 
    def forward(self, tensor_1, tensor_2):
        intermediate = torch.matmul(tensor_1, self.weight_matrix)
        result = (intermediate * tensor_2).sum(dim=-1) + self.bias
        if self.activation is not None:
            result = self.activation(result)
        return result
    
    
class TriLinearSimilarity(nn.Module):
 
    def __init__(self, input_dim, activation=None):
        super(TriLinearSimilarity, self).__init__()
        self.weight_vector = nn.Parameter(torch.Tensor(3 * input_dim)).to(device)
        self.bias = nn.Parameter(torch.Tensor(1)).to(device)
        self.activation = activation
        self.reset_parameters()
 
    def reset_parameters(self):
        std = math.sqrt(6 / (self.weight_vector.size(0) + 1))
        self.weight_vector.data.uniform_(-std, std)
        self.bias.data.fill_(0)
 
    def forward(self, tensor_1, tensor_2):
        combined_tensors = torch.cat([tensor_1, tensor_2, tensor_1 * tensor_2], dim=-1)
        result = torch.matmul(combined_tensors, self.weight_vector) + self.bias
        if self.activation is not None:
            result = self.activation(result)
        return result
    

# class TriLinearSimilarity(nn.Module):
 
#     def __init__(self, input_dim, activation=None):
#         super(TriLinearSimilarity, self).__init__()
#         self.Final_layer = nn.Linear(input_dim, 1).to(device)
#         self.hidden_size = input_dim
#         self.DNN = nn.Linear(4 * input_dim, input_dim).to(device) 
#         self.activation = activation
#         self.reset_parameters()
 
#     def reset_parameters(self):
#         stdv = 1.0 / self.hidden_size ** 0.5
#         for weight in self.parameters():
#             nn.init.uniform_(weight, -stdv, stdv)
 
#     def forward(self, tensor_1, tensor_2):
#         combined_tensors = torch.cat([tensor_1, tensor_2, tensor_1 * tensor_2, tensor_1 - tensor_2], dim=-1)
#         combined_tensors = torch.relu(self.DNN(combined_tensors))
#         result = self.Final_layer(combined_tensors).squeeze()  # B, L
#         if self.activation is not None:
#             result = self.activation(result)
#         return result
    
    
def Generate_Instance_Matrix(all_sessions):
    data, edge = [], []
    for j in range(len(all_sessions)):
        session = np.unique(all_sessions[j]) - 1
        data.extend(session)
        length = len(session)
        edge.extend([j] * length)
    H = torch.cat([torch.tensor(data).view(1, -1), torch.tensor(edge).view(1, -1)], dim=0)
    # print("H: ", H.shape)
    return H


def data_mask(all_sessions, n_node):
    """
    目前没使用这个函数。
    这里由于要考虑padding, 所以n_node = item_num + 1
    最后的矩阵就是[B * n_node]的, 是H.T
    """
    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(all_sessions)):
        session = np.unique(all_sessions[j])
        if session[0] == 0:
            session = np.delete(session, 0)
        length = len(session)
        s = indptr[-1]
        indptr.append((s + length))
        data.extend([1] * length)
        indices.extend(session)
    indices = np.array(indices) - 1
    matrix = csr_matrix((data, indices, indptr), shape=(len(all_sessions), n_node))

    return matrix


def get_DB(data, n_node):
    """
    目前没使用这个函数。
    返回B^(-1)和D^(-1)
    H_T.sum(axis=1)和H.sum(axis=1)分别表示了边和点的度，分别对应B和D。
    """
    H_T, indices, indptr = data_mask(data, n_node)
    B = 1.0 / H_T.sum(axis=1).reshape(1, -1)
    D = 1.0 / H_T.T.sum(axis=1).reshape(1, -1)
    return B, D, indices, indptr


def shift_data_padding(data_padding):
    """
    由于没用H矩阵，所以没使用这个函数。
    将0全变成1,(用mask保证不计算入attention)，所有Item编号-1，转换为tensor格式
    """
    tmp = torch.from_numpy(data_padding) - 1
    tmp[tmp < 0] = 0
    return tmp.long()


def get_adjancency(H_T):
    """
    目前没使用这个函数。
    给出H矩阵，返回D^(-1)HB^(-1)H^T
    """
    BH_T = H_T.T.multiply(1.0 / H_T.sum(axis=1).reshape(1, -1))
    BH_T = BH_T.T # (M, N)
    H = H_T.T  # 就是 incidence matrix， N * M
    DH = H.T.multiply(1.0 / H.sum(axis=1).reshape(1, -1))
    DH = DH.T # (N, M)
    DHBH_T = np.dot(DH, BH_T)
    adjacency = DHBH_T.tocoo()  # .tocoo() 将稠密矩阵转为稀疏矩阵。
    values = adjacency.data  # D^(-1)HB^(-1)H^T
    indices = np.vstack((adjacency.row, adjacency.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = adjacency.shape
    adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return adjacency


def generate_neg(num, mx, s=None):
    """
    :param num: 要生成多少个
    :param mx: 最大要几
    :param s: 集合，生成的不要在s中
    :return: 生成的负采样
    """
    xx = []
    for i in range(num):
        t = np.random.randint(1, mx)
        if s:
            while t in s:
                t = np.random.randint(1, mx)
        else:
            xx.append(t)
    return torch.tensor(xx)


def find_k_largest(K, candidates):
    n_candidates = []
    for iid, score in enumerate(candidates[:K]):
        n_candidates.append((iid, score))
    n_candidates.sort(key=lambda d: d[1], reverse=True)
    k_largest_scores = [item[1] for item in n_candidates]
    ids = [item[0] for item in n_candidates]
    # find the N biggest scores
    for iid, score in enumerate(candidates):
        ind = K
        l = 0
        r = K - 1
        if k_largest_scores[r] < score:
            while r >= l:
                mid = int((r - l) / 2) + l
                if k_largest_scores[mid] >= score:
                    l = mid + 1
                elif k_largest_scores[mid] < score:
                    r = mid - 1
                if r < l:
                    ind = r
                    break
        # move the items backwards
        if ind < K - 2:
            k_largest_scores[ind + 2:] = k_largest_scores[ind + 1:-1]
            ids[ind + 2:] = ids[ind + 1:-1]
        if ind < K - 1:
            k_largest_scores[ind + 1] = score
            ids[ind + 1] = iid
    return ids  # ,k_largest_scores


class SessionDataSet(Dataset):
    """
    history_session, current_session, user -> label
    """
    def __init__(self, DATA, history_session=None, current_session=None, user_session=None, label=None, mask=None):
        super(SessionDataSet, self).__init__()
        self.history_session, self.current_session, self.user_session, self.label, self.mask = [], [], [], [], []
        '''
        如果外面弄好了就传进来。否则就在里面自己弄
        '''
        if history_session:
            self.history_session = np.array(history_session)
            self.current_session = np.array(current_session)
            self.user_session = np.array(user_session)
            self.label = np.array(label)
            self.mask = np.array(mask)
        else:
            for (h, c, u, l) in DATA:
                tmp = np.stack(h)
                self.history_session.append(tmp)
                self.current_session.append(c)
                self.user_session.append(u)
                self.label.append(l)
                mask = np.count_nonzero(tmp, axis=1)
                mask = (mask == 0)
                self.mask.append(mask)
            self.history_session = np.array(self.history_session)
            self.current_session = np.array(self.current_session)
            self.user_session = np.array(self.user_session)
            self.label = np.array(self.label)
            self.mask = np.array(self.mask)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return self.history_session[item], self.current_session[item], self.user_session[item], self.label[item], self.mask[item]


class SessionDataSet_Single(Dataset):
    """
    独立session
    """
    def __init__(self, DATA):
        super(SessionDataSet_Single, self).__init__()
        self.session, self.label = DATA
        self.mask = []
        for session in self.session:
            mask = (session == 0)
            self.mask.append(mask)
        self.label = np.array(self.label)
        self.mask = np.array(self.mask)
      #  print(type(self.session))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return self.session[item], self.label[item], self.mask[item]
    
    
class SessionDataSet_Single_User(Dataset):
    """
    独立session
    """
    def __init__(self, DATA):
        super(SessionDataSet_Single_User, self).__init__()
        self.session, self.label, self.user = DATA
        self.mask = []
        for session in self.session:
            mask = (session == 0)
            self.mask.append(mask)
        self.label = np.array(self.label)
        self.mask = np.array(self.mask)
        self.user = np.array(self.user)
      #  print(type(self.session))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return self.session[item], self.label[item], self.mask[item], self.user[item]


def pad_sequences(all_seq, maxlen=20):
    """
    防止某些sb服务器没有keras
    """
    for x in all_seq:
        ned = maxlen - len(x)
        for _ in range(ned):
            x.insert(0, 0)
    all_seq = np.array(all_seq)
    return all_seq









