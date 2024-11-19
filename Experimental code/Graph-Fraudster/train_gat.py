# train_gat.py

import argparse
import os
import time

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from itertools import chain
import random
from sklearn import metrics
from datasets import load_data1
import data_split
from utils import normalize_adj_tensor

# 参数解析与环境设置
parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='pubmed', help='Dataset to train')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--epoches', type=int, default=200, help='Number of epochs')
parser.add_argument('--hidden_dim', type=int, default=8, help='Hidden layer dimensions')
parser.add_argument('--nheads', type=int, default=2, help='Number of attention heads')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--seed', type=int, default=30, help='Random seed.')
parser.add_argument('--p', type=float, default=0.5, help='Split ratio for data graph A')


args = parser.parse_args()

# 设置随机种子和设备
args.cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# 加载数据集和图结构
adj, features, idx_train, idx_val, idx_test, labels = load_data1(args.datasets)
labels = torch.LongTensor(labels)  # 转换为 Tensor
A_A, A_B, X_A, X_B = data_split.split_graph(args, adj, features, split_method='com', with_s=True, with_f=True)
X_A, X_B = torch.FloatTensor(X_A.todense()).to(device), torch.FloatTensor(X_B.todense()).to(device)
A_A, A_B = normalize_adj_tensor(torch.FloatTensor(A_A.todense()).to(device)), normalize_adj_tensor(torch.FloatTensor(A_B.todense()).to(device))



# 定义 GAT 层和模型结构
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)
        N = h.size()[0]
        a_input = self._prepare_attentional_mechanism_input(h)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = torch.nn.functional.softmax(attention, dim=1)
        attention = torch.nn.functional.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        return torch.nn.functional.elu(h_prime) if self.concat else h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]
        Wh_repeat = Wh.repeat_interleave(N, dim=0)
        Wh_repeat_interleave = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeat, Wh_repeat_interleave], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = nn.ModuleList(
            [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        )
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return torch.nn.functional.log_softmax(x, dim=1)

# 定义客户端和服务器模型
nheads = args.nheads
hidden_dim = args.hidden_dim
client_output_dim = hidden_dim * nheads
nclass = int(labels.max()) + 1
models = {
    "client_A": GAT(nfeat=X_A.shape[1], nhid=args.hidden_dim, nclass=client_output_dim, dropout=args.dropout, alpha=0.2, nheads=nheads).to(device),
    "client_B": GAT(nfeat=X_B.shape[1], nhid=args.hidden_dim, nclass=client_output_dim, dropout=args.dropout, alpha=0.2, nheads=nheads).to(device),
    "server": nn.Sequential(nn.Linear(2 * client_output_dim, nclass)).to(device)
}

# 训练函数
def train(XA, AA, XB, AB, idx_train, labels):
    optimizer = optim.Adam(params=chain(models['client_A'].parameters(), models['client_B'].parameters(), models['server'].parameters()), lr=args.lr, weight_decay=5e-4)
    for epoch in range(args.epoches):
        models['client_A'].train(), models['client_B'].train(), models['server'].train()
        optimizer.zero_grad()
        client_out_A, client_out_B = models['client_A'](XA, AA), models['client_B'](XB, AB)
        server_input = torch.cat((client_out_A, client_out_B), dim=1)
        pred = models['server'](server_input)
        pred = torch.log_softmax(pred, dim=-1)
        loss = nn.NLLLoss()(pred[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()
        if epoch % 1 == 0:
            print("Epoch: %d, Loss: %f" % (epoch, loss.cpu().item()))
            print(time.ctime())
    return client_out_A.detach(), client_out_B.detach(), pred.detach()

# 训练模型并保存
emb_A, emb_B, pred_S = train(X_A, A_A, X_B, A_B, idx_train, labels)
model_path = f'save_model/main/GAT/{args.datasets}{args.seed}'
os.makedirs(model_path, exist_ok=True)
torch.save(models['client_A'].state_dict(), f'{model_path}/A.pkl')
torch.save(models['client_B'].state_dict(), f'{model_path}/B.pkl')
torch.save(models['server'].state_dict(), f'{model_path}/server.pkl')
torch.save(emb_A, f'{model_path}/emb_A.pth')
torch.save(emb_B, f'{model_path}/emb_B.pth')

# 保存 labels_B.pth
torch.save(pred_S.max(1)[1], f'{model_path}/labels_B.pth')

# 定义并保存 surrogate_server
surrogate_server = nn.Sequential(nn.Linear(2 * client_output_dim, nclass)).to(device)
surrogate_server.load_state_dict(models['server'].state_dict())
torch.save(surrogate_server.state_dict(), f'{model_path}/surrogate_server.pkl')

print('Training complete and models saved.')

