# attack_gat.py

import torch
import torch.nn as nn
import numpy as np
import argparse
from datasets import load_data1
import data_split
from utils import normalize_adj_tensor
import os
import warnings

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

# 忽略 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# 参数解析
attack_size = 100  # 攻击比例
parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='citeseer', help='Dataset to attack')
parser.add_argument('--seed', type=int, default=30, help='Random seed')
parser.add_argument('--eps', type=float, default=0.02, help='Epsilon for FGSM attack')  # 调低 eps
parser.add_argument('--hidden_dim', type=int, default=8, help='Hidden layer dimensions')  # 设为8
parser.add_argument('--nheads', type=int, default=2, help='Number of attention heads')    # 设为2
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu')
parser.add_argument('--p', type=float, default=0.5, help='Split ratio for data graph A')
parser.add_argument('--q', type=float, default=0.5, help='Split ratio for data graph B')
parser.add_argument('--save_path', type=str, default='adversarial_samples.npy', help='Path to save adversarial samples')  # 新增参数
args = parser.parse_args()

# 设备设置
args.cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# 加载数据和分割
adj, features, idx_train, idx_val, idx_test, labels = load_data1(args.datasets)
labels = torch.LongTensor(labels).to(device)

A_A, A_B, X_A, X_B = data_split.split_graph(args, adj, features, split_method='com', with_s=True, with_f=True)
X_A = torch.FloatTensor(X_A.todense()).to(device)
X_B = torch.FloatTensor(X_B.todense()).to(device)
A_A = normalize_adj_tensor(torch.FloatTensor(A_A.todense())).to(device)
A_B = normalize_adj_tensor(torch.FloatTensor(A_B.todense())).to(device)

# 转换 idx_test 为 torch.Tensor
idx_test = torch.LongTensor(idx_test).to(device)

# 定义客户端和服务器模型
nheads = args.nheads
hidden_dim = args.hidden_dim
client_output_dim = hidden_dim * nheads
nclass = int(labels.max()) + 1

client_A = GAT(nfeat=X_A.shape[1], nhid=hidden_dim, nclass=client_output_dim, dropout=args.dropout, alpha=args.alpha, nheads=nheads).to(device)
client_B = GAT(nfeat=X_B.shape[1], nhid=hidden_dim, nclass=client_output_dim, dropout=args.dropout, alpha=args.alpha, nheads=nheads).to(device)
server = nn.Sequential(
    nn.Linear(2 * client_output_dim, nclass)
).to(device)

# 加载训练好的模型
model_path = f'save_model/main/GAT/{args.datasets}{args.seed}'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model path {model_path} does not exist.")

client_A.load_state_dict(torch.load(f'{model_path}/A.pkl', map_location=device))
client_B.load_state_dict(torch.load(f'{model_path}/B.pkl', map_location=device))
server.load_state_dict(torch.load(f'{model_path}/server.pkl', map_location=device))
client_A.eval()
client_B.eval()
server.eval()

# 定义 surrogate_server 模型（与 server 相同的结构）
surrogate_server = nn.Sequential(
    nn.Linear(2 * client_output_dim, nclass)
).to(device)
surrogate_server.load_state_dict(torch.load(f'{model_path}/surrogate_server.pkl', map_location=device))
surrogate_server.eval()

# 加载 labels_B.pth
labels_B_path = f'{model_path}/labels_B.pth'
if not os.path.exists(labels_B_path):
    raise FileNotFoundError(f"Labels path {labels_B_path} does not exist.")
labels_B = torch.load(labels_B_path).to(device)

# FGSM 攻击函数
def fgsm_attack(XA, AA, XB, AB, client_A, client_B, surrogate_server, labels, target_node, eps=0.05):
    # 获取可学习的输入特征副本
    XA_adv = XA.clone().detach()
    XA_adv.requires_grad = True

    # 前向传播
    client_out_A = client_A(XA_adv, AA)
    client_out_B = client_B(XB, AB)
    cat_input = torch.cat((client_out_A[target_node], client_out_B[target_node]), dim=-1)

    pred = surrogate_server(cat_input)
    pred = torch.log_softmax(pred, dim=-1)
    loss = nn.NLLLoss()(pred.unsqueeze(0), labels[target_node].unsqueeze(0))

    # 反向传播
    surrogate_server.zero_grad()
    loss.backward()

    # 获取目标节点的梯度
    data_grad = XA_adv.grad.data[target_node]
    sign_data_grad = data_grad.sign()

    # 生成对抗特征
    perturbed_features = XA_adv[target_node] + eps * sign_data_grad

    # 对特征进行裁剪（根据特征范围调整）
    perturbed_features = torch.clamp(perturbed_features, XA.min(), XA.max())

    return perturbed_features

# 评估函数
def evaluate(XA, AA, XB, AB, client_A, client_B, server, labels, target_node, XA_adv=None):
    client_A.eval()
    client_B.eval()
    server.eval()

    with torch.no_grad():
        if XA_adv is not None:
            # 使用对抗特征进行前向传播
            client_out_A = client_A(XA_adv, AA)
        else:
            # 使用正常特征进行前向传播
            client_out_A = client_A(XA, AA)

        client_out_B = client_B(XB, AB)

        # 获取目标节点的嵌入
        client_out_A_node = client_out_A[target_node].unsqueeze(0)
        client_out_B_node = client_out_B[target_node].unsqueeze(0)

        # 拼接嵌入并进行预测
        server_input = torch.cat((client_out_A_node, client_out_B_node), dim=1)
        pred = server(server_input)
        pred = torch.log_softmax(pred, dim=1)

        # 获取真实标签和预测标签
        target_label = labels[target_node].unsqueeze(0)
        predicted_label = pred.argmax(dim=1)
        accuracy = (predicted_label == target_label).float().item()

        # 打印预测信息
        print(f"Predicted: {predicted_label.item()}, True label: {target_label.item()}")

        return accuracy

# 攻击和评估
target_nodes = np.random.choice(idx_test.cpu().numpy(), size=attack_size, replace=False)
correct_before = 0
correct_after = 0
total = len(target_nodes)

# Initialize XA_adv as a clone of X_A (稠密张量)
XA_adv = X_A.clone().detach()

for target_node in target_nodes:
    # 攻击前准确率
    acc_before_attack = evaluate(XA_adv, A_A, X_B, A_B, client_A, client_B, server, labels_B, target_node)

    # 生成对抗特征并进行攻击
    perturbed_features = fgsm_attack(XA_adv, A_A, X_B, A_B, client_A, client_B, surrogate_server, labels_B, target_node, eps=args.eps)

    # 更新 XA_adv
    XA_adv[target_node] = perturbed_features.detach()  # 使用 detach() 确保不需要梯度

    # 攻击后准确率
    acc_after_attack = evaluate(XA_adv, A_A, X_B, A_B, client_A, client_B, server, labels_B, target_node, XA_adv=XA_adv)

    correct_before += acc_before_attack
    correct_after += acc_after_attack

    # 输出攻击前后准确率变化
    print(f"Node {target_node}: Accuracy before attack: {acc_before_attack:.4f}, Accuracy after attack: {acc_after_attack:.4f}")

# 保存对抗样本
XA_adv_numpy = XA_adv.detach().cpu().numpy()  # 使用 detach() 分离计算图
save_path = os.path.join(model_path, args.save_path)
np.save(save_path, XA_adv_numpy)
print(f"Adversarial samples saved to {save_path}")

# 输出整体准确率
overall_acc_before = correct_before / total
overall_acc_after = correct_after / total

print(f"Overall Accuracy before attack: {overall_acc_before:.4f}")
print(f"Overall Accuracy after attack: {overall_acc_after:.4f}")
