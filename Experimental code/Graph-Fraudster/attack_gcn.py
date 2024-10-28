# attack_gcn.py

import argparse
import torch
import torch.nn as nn
import numpy as np
from utils import normalize_adj, sparse_mx_to_torch_sparse_tensor
from datasets import load_data1
import data_split
from gcn import GCN
import os
import warnings

# 忽略 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# 参数解析
attack_size = 100
parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='cora', help='Dataset to attack')
parser.add_argument('--seed', type=int, default=30, help='Random seed')
parser.add_argument('--eps', type=float, default=0.02, help='Epsilon for FGSM attack')  # 调低eps以减少扰动强度
parser.add_argument('--hidden_dim', type=int, default=32, help='Dimensions of hidden layers')
parser.add_argument('--out_dim', type=int, default=16, help='Dimensions of out layers')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability)')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight for l2 loss on embedding matrix')
parser.add_argument('--p', type=float, default=0.5, help='Hold data p.')
parser.add_argument('--q', type=float, default=0.5, help='Hold data q.')
parser.add_argument('--save_path', type=str, default='adversarial_samples.npy',
                    help='Path to save adversarial samples')  # 新增参数
args = parser.parse_args()

# 设置随机种子
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

# 环境设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = f'save_model/main/GCN/{args.datasets}{args.seed}'

# 检查模型路径是否存在
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model path {model_path} does not exist.")

# 加载数据和模型
adj, features, idx_train, idx_val, idx_test, labels = load_data1(args.datasets)
labels = torch.tensor(labels, dtype=torch.long, device=device)  # 确保labels为张量

A_A, A_B, X_A, X_B = data_split.split_graph(args, adj, features, split_method='com', with_s=True, with_f=True)
A_A = normalize_adj(A_A)
A_B = normalize_adj(A_B)
A_A = sparse_mx_to_torch_sparse_tensor(A_A).to(device)
A_B = sparse_mx_to_torch_sparse_tensor(A_B).to(device)

# **转换特征张量为稠密格式**
X_A = torch.FloatTensor(X_A.toarray()).to(device)
X_B = torch.FloatTensor(X_B.toarray()).to(device)

idx_test = torch.LongTensor(idx_test).to(device)

# 定义 client_A 和 client_B 模型实例，并加载其 state_dict
client_A = GCN(nfeat=X_A.shape[1], nclass=args.out_dim, nhid=args.hidden_dim, device=device, dropout=0).to(device)
client_A.load_state_dict(torch.load(f'{model_path}/A.pkl', map_location=device))

client_B = GCN(nfeat=X_B.shape[1], nclass=args.out_dim, nhid=args.hidden_dim, device=device, dropout=0).to(device)
client_B.load_state_dict(torch.load(f'{model_path}/B.pkl', map_location=device))

# 动态确定 server 模型的输入维度
with torch.no_grad():
    client_out_A = client_A(X_A, A_A)
    client_out_B = client_B(X_B, A_B)
    server_input_dim = client_out_A.shape[1] + client_out_B.shape[1]

server = nn.Sequential(
    nn.Linear(server_input_dim, int(labels.max() + 1))
).to(device)
server.load_state_dict(torch.load(f'{model_path}/server.pkl', map_location=device))
server.eval()  # 确保server处于评估模式


# 定义 FGSM 攻击函数 (对输入特征进行攻击)
def fgsm_attack(XA, AA, XB, AB, client_A, client_B, server, labels, target_node, eps):
    # Clone XA and set requires_grad
    XA_adv = XA.clone().detach()
    XA_adv.requires_grad = True

    # Forward pass
    client_out_A = client_A(XA_adv, AA)
    client_out_B = client_B(XB, AB)

    # Concatenate embeddings for target node
    server_input = torch.cat((client_out_A[target_node].unsqueeze(0), client_out_B[target_node].unsqueeze(0)), dim=1)

    # Compute loss
    pred = server(server_input)
    pred = torch.log_softmax(pred, dim=1)
    loss = torch.nn.functional.nll_loss(pred, labels[target_node].unsqueeze(0))

    # Backward pass
    loss.backward()

    # Get gradient sign
    grad_sign = XA_adv.grad[target_node].sign()

    # Perturb the features
    XA_adv = XA_adv.clone().detach()
    XA_adv[target_node] = XA_adv[target_node] + eps * grad_sign

    # Optionally clip the features to valid range
    # XA_adv = torch.clamp(XA_adv, 0, 1)

    # Calculate difference norm
    diff = (XA_adv[target_node] - XA[target_node]).norm().item()
    print(f"Embedding difference norm: {diff}")

    return XA_adv[target_node]


# 定义评估函数
def evaluate(XA, AA, XB, AB, client_A, client_B, server, target_node, labels, XA_adv=None):
    client_A.eval()
    client_B.eval()
    server.eval()
    with torch.no_grad():
        if XA_adv is not None:
            # 使用对抗扰动后的嵌入
            XA_new = XA.clone().detach()
            XA_new[target_node] = XA_adv
            client_out_A = client_A(XA_new, AA)
            client_out_B = client_B(XB, AB)
        else:
            # 使用正常的嵌入
            client_out_A = client_A(XA, AA)
            client_out_B = client_B(XB, AB)

        # 获取目标节点的嵌入
        client_out_A_node = client_out_A[target_node].unsqueeze(0)  # 维度 [1, 16]
        client_out_B_node = client_out_B[target_node].unsqueeze(0)  # 维度 [1, 16]

        # 打印调试信息
        print("Shape of client_out_A:", client_out_A_node.shape)  # 应该是 [1, 16]
        print("Shape of client_out_B:", client_out_B_node.shape)  # 应该是 [1, 16]

        server_input = torch.cat((client_out_A_node, client_out_B_node), dim=1)
        print("Shape of server_input:", server_input.shape)  # 应该是 [1, 32]

        pred = server(server_input)
        pred = torch.log_softmax(pred, dim=1)

        target_label = labels[target_node].unsqueeze(0)  # 确保是张量
        predicted_label = pred.argmax(dim=1)
        accuracy = (predicted_label == target_label).float().item()

        print(f"Predicted: {predicted_label.item()}, True label: {target_label.item()}")

        return accuracy


# 动态创建 surrogate_server 模型，并加载其 state_dict
surrogate_server = nn.Sequential(
    nn.Linear(server_input_dim, int(labels.max() + 1))
).to(device)
surrogate_server.load_state_dict(torch.load(f'{model_path}/surrogate_server.pkl', map_location=device))
surrogate_server.eval()

# 攻击和评估
target_nodes = np.random.choice(idx_test.cpu().numpy(), size=attack_size, replace=False)
correct_before = 0
correct_after = 0
total = len(target_nodes)

# Initialize XA_adv as a clone of X_A (稠密张量)
XA_adv = X_A.clone().detach()

for target_node in target_nodes:
    print(f"\nAttacking node {target_node}...")

    # Evaluate before attack
    acc_before_attack = evaluate(XA_adv, A_A, X_B, A_B, client_A, client_B, server, target_node, labels)

    # Perform FGSM attack
    XA_adv_node = fgsm_attack(XA_adv, A_A, X_B, A_B, client_A, client_B, surrogate_server, labels, target_node,
                              eps=args.eps)

    # Update XA_adv
    XA_adv[target_node] = XA_adv_node

    # Evaluate after attack
    acc_after_attack = evaluate(XA_adv, A_A, X_B, A_B, client_A, client_B, server, target_node, labels, XA_adv_node)

    correct_before += acc_before_attack
    correct_after += acc_after_attack

    print(
        f"Node {target_node}: Accuracy before attack: {acc_before_attack:.4f}, Accuracy after attack: {acc_after_attack:.4f}")

# 保存对抗样本
XA_adv_numpy = XA_adv.cpu().numpy()
save_path = os.path.join(model_path, args.save_path)
np.save(save_path, XA_adv_numpy)
print(f"\nAdversarial samples saved to {save_path}")

# 输出整体准确率
overall_acc_before = correct_before / total
overall_acc_after = correct_after / total

print(f"\nOverall Accuracy before attack: {overall_acc_before:.4f}")
print(f"Overall Accuracy after attack: {overall_acc_after:.4f}")
