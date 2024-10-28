# train_gat.py

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from itertools import chain
from datasets import load_data1
from utils import normalize_adj, sparse_mx_to_torch_sparse_tensor
import data_split
from gcn import GCN

# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='pubmed', help='Dataset to train, citeseer')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
parser.add_argument('--epoches', type=int, default=200, help='Number of training epochs')
parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden layer dimensions')
parser.add_argument('--out_dim', type=int, default=16, help='Output layer dimensions')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight for L2 loss')
parser.add_argument('--seed', type=int, default=30, help='Random seed')
parser.add_argument('--p', type=float, default=0.5, help='Hold data p')
args = parser.parse_args()

# 环境设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# 加载数据
adj, features, idx_train, idx_val, idx_test, labels = load_data1(args.datasets)
labels = torch.LongTensor(labels)  # 转换为 Tensor
idx_train, idx_val, idx_test = map(torch.LongTensor, [idx_train, idx_val, idx_test])

A_A, A_B, X_A, X_B = data_split.split_graph(args, adj, features, split_method='com', with_s=True, with_f=True)

# 转换为稀疏张量
A_A = sparse_mx_to_torch_sparse_tensor(normalize_adj(A_A)).to(device)
A_B = sparse_mx_to_torch_sparse_tensor(normalize_adj(A_B)).to(device)
X_A = sparse_mx_to_torch_sparse_tensor(X_A).to(device)
X_B = sparse_mx_to_torch_sparse_tensor(X_B).to(device)

# 初始化模型
models = {
    "client_A": GCN(nfeat=X_A.shape[1], nclass=args.out_dim, nhid=args.hidden_dim, device=device, dropout=args.dropout),
    "client_B": GCN(nfeat=X_B.shape[1], nclass=args.out_dim, nhid=args.hidden_dim, device=device, dropout=args.dropout),
    "server": nn.Sequential(nn.Linear(2 * args.out_dim, int(labels.max() + 1))),
}

for model in models.values():
    model.to(device)

# 定义训练函数
def train(XA, AA, XB, AB):
    optimizer = optim.Adam(
        params=chain(models['client_A'].parameters(), models['client_B'].parameters(), models['server'].parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    for epoch in range(args.epoches):
        models['client_A'].train(), models['client_B'].train(), models['server'].train()
        optimizer.zero_grad()

        client_out_A = models['client_A'](XA, AA)
        client_out_B = models['client_B'](XB, AB)
        server_input = torch.cat((client_out_A, client_out_B), dim=1)
        pred = models['server'](server_input)
        pred = torch.log_softmax(pred, dim=-1)

        loss = nn.NLLLoss()(pred[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, train loss: {loss.item()}")

    return client_out_A, client_out_B, pred

# 训练并保存主模型
emb_A, emb_B, pred_S = train(X_A, A_A, X_B, A_B)
model_path = f'save_model/main/GCN/{args.datasets}{args.seed}/'
os.makedirs(model_path, exist_ok=True)

torch.save(models['client_A'].state_dict(), f'{model_path}/A.pkl')
torch.save(models['client_B'].state_dict(), f'{model_path}/B.pkl')
torch.save(models['server'].state_dict(), f'{model_path}/server.pkl')
torch.save(emb_A, f'{model_path}/emb_A.pth')
torch.save(emb_B, f'{model_path}/emb_B.pth')
torch.save(pred_S.max(1)[1], f'{model_path}/labels_B.pth')

# 定义 surrogate_server 模型
surrogate_server = nn.Sequential(
    nn.Linear(2 * args.out_dim, int(labels.max() + 1))
).to(device)

# surrogate_server 训练并保存
surrogate_optimizer = optim.Adam(surrogate_server.parameters(), lr=args.lr, weight_decay=args.weight_decay)
for epoch in range(args.epoches):
    surrogate_server.train()
    surrogate_optimizer.zero_grad()
    # 对 emb_A 和 emb_B 进行 detach
    pred = surrogate_server(torch.cat((emb_A.detach(), emb_B.detach()), dim=1))
    pred = torch.log_softmax(pred, dim=-1)
    surrogate_loss = nn.NLLLoss()(pred[idx_train], labels[idx_train])
    surrogate_loss.backward()
    surrogate_optimizer.step()

    if epoch % 10 == 0:
        print(f"Surrogate Epoch: {epoch}, surrogate loss: {surrogate_loss.item()}")

torch.save(surrogate_server.state_dict(), f'{model_path}/surrogate_server.pkl')
print("Training complete and models saved.")
