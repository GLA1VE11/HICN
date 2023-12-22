import random
import os
import datetime
import argparse
import pickle
from utils import *
import copy
import importlib

device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='RetailRocket', help='dataset name: RetailRocket/Tmall/yoo64')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--hidden_dim', type=int, default=100, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--layers', type=int, default=2, help='the number of layer used')
parser.add_argument('--sample_num', type=int, default=10, help='sample session num')  
parser.add_argument('--num_heads', type=int, default=4, help='num_heads for attn')
parser.add_argument('--use_gnn', default='hgat', help='use gnn or not')
opt = parser.parse_args()
print(opt)

# 根据条件动态导入模块
if opt.dataset == 'Tmall':
    HICN_module = importlib.import_module('models.Tmall')
else:
    HICN_module = importlib.import_module('models.Normal')


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    f = open(f"result.txt", "a")
    # sys.stdout = f
    # sys.stderr = f
    setup_seed(2023)
    with open(f"./datasets/{opt.dataset}_session.txt", "rb") as f1:
        DATA = pickle.load(f1)
    with open(f"./datasets/{opt.dataset}_neighbour.txt", "rb") as f1:
        DATA1 = pickle.load(f1)
    with open(f'./datasets/{opt.dataset}_train.txt', "rb") as f1:
        DATA2 = pickle.load(f1)
    with open(f'./datasets/{opt.dataset}_test.txt', "rb") as f1:
        DATA3 = pickle.load(f1)
    # all_seq是从1下标开始的
    '''
    all_seq 对应每个session的item 信息。 
    item_session 是一个dict, 记录每个item 对应的session的id。
    max_len 是每个session最长的长度
    '''
    if opt.dataset in ['Tmall', 'yoo64', 'RetailRocket',] :
        all_seq, item_session, item_num, max_len = DATA  # 是用于没有user的
    else: 
        all_seq, user_seq, item_session, user_num, item_num, max_len = DATA # 用于有user的
    print(max_len, item_num)
    if opt.use_gnn == 'hgat':
        print('Construct Hpergraph: ', datetime.datetime.now())
        tmp = copy.deepcopy(all_seq)
        data_padding = pad_sequences(tmp, maxlen=max_len)
        hyperedge_index = Generate_Instance_Matrix(all_seq)
        H_T = data_mask(all_seq, item_num) # len(all_seq), item_num
        adj = get_adjancency(H_T)
        print(hyperedge_index.shape)
    print('Construct Dataset: ', datetime.datetime.now())
    if opt.dataset in ['Tmall', 'yoo64', 'RetailRocket',] :
        dataset = SessionDataSet_Single(DATA2)
        dataset_test = SessionDataSet_Single(DATA3)
    else:
        dataset = SessionDataSet_Single_User(DATA2)
        dataset_test = SessionDataSet_Single_User(DATA3)
    
    print('Construct Done! ', datetime.datetime.now())
    train_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=False, drop_last=True)  # Here

    if opt.dataset in ['Tmall', 'yoo64']:
        nei_num = 10
    else:
        nei_num = 5
    model = HICN_module.HICN(opt.dataset, item_num, max_len, opt.hidden_dim, opt.lr, opt.num_heads,
                     nei_num, DATA1, opt.batch_size, all_seq, item_session, opt.sample_num, opt.layers, adj, hyperedge_index).to(device)
    PATH = f'./checkpoint/{opt.dataset}_best.pt'
    # model.load_state_dict(torch.load(PATH))
    f.flush()
    HICN_module.train_epoch(opt.epoch, model, train_loader, test_loader, item_num, f, opt.dataset)


if __name__ == '__main__':
    main()
