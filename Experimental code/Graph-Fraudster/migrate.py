# migrate.py
from email.policy import default

import torch
import torch.nn as nn
import numpy as np
import argparse
from datasets import load_data1
import data_split
from utils import normalize_adj_tensor, sparse_mx_to_torch_sparse_tensor
import os
import warnings

# 导入模型类
from gcn import GCN
from gat import GAT

def main():
    # 忽略 FutureWarning
    warnings.filterwarnings("ignore", category=FutureWarning)

    # 参数解析
    migrate_model = "GCN"  # 源，指对抗网络
    model = "GAT" # 目标，指保存的模型
    migrate_dataset = "cora" # 源，指对抗网络
    dataset = "citeseer" # 目标，指保存的模型

    parser = argparse.ArgumentParser(description='Migrate and evaluate GCN or GAT model with adversarial samples.')
    parser.add_argument('--model_type', default=f"{model}", type=str, choices=['GCN', 'GAT'],
                        help='Type of the model: GCN or GAT')
    parser.add_argument('--datasets', default=f"{dataset}", type=str, help='Dataset to use, e.g., citeseer, pubmed')
    parser.add_argument('--seed', type=int, default=30, help='Random seed')
    parser.add_argument('--hidden_dim', type=int, help='Hidden layer dimensions (automatically set based on model type if not provided)')
    parser.add_argument('--nheads', type=int, default=2, help='Number of attention heads (only for GAT)')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for leaky_relu (only for GAT)')
    parser.add_argument('--p', type=float, default=0.5, help='Split ratio for data graph A')
    parser.add_argument('--q', type=float, default=0.5, help='Split ratio for data graph B')
    parser.add_argument('--model_path', default=f"save_model/main/{model}/{dataset}30", type=str, help='Path to the saved model directory')
    parser.add_argument('--adversarial_path', type=str, default=f'save_model/main/{migrate_model}/{migrate_dataset}30/adversarial_samples_to_{dataset}.npy',
                        help='Path to adversarial samples')
    parser.add_argument('--labels_B_path', default=f"save_model/main/{model}/{dataset}30/labels_B.pth",type=str, help='Path to labels_B.pth (only for GAT)')
    args = parser.parse_args()

    # 验证参数
    if args.model_type == 'GAT' and args.labels_B_path is None:
        parser.error("--labels_B_path is required when model_type is GAT")

    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 动态设置 hidden_dim 基于 model_type
    if args.hidden_dim is None:
        if args.model_type == 'GCN':
            args.hidden_dim = 32  # GCN 的 hidden_dim
        elif args.model_type == 'GAT':
            args.hidden_dim = 8   # GAT 的 hidden_dim

    # 模型和对抗样本路径封装
    model_path = args.model_path
    adversarial_path = os.path.join(args.adversarial_path)

    # 检查路径是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist.")
    if not os.path.exists(adversarial_path):
        raise FileNotFoundError(f"Adversarial samples path {adversarial_path} does not exist.")

    # 对于 GAT，加载 labels_B.pth
    if args.model_type == 'GAT':
        if not os.path.exists(args.labels_B_path):
            raise FileNotFoundError(f"Labels path {args.labels_B_path} does not exist.")

    # 加载数据和分割
    adj, features, idx_train, idx_val, idx_test, labels = load_data1(args.datasets)
    labels = torch.LongTensor(labels).to(device)

    # 分割图
    A_A, A_B, X_A, X_B = data_split.split_graph(args, adj, features, split_method='com', with_s=True, with_f=True)

    # 转换特征张量为稠密格式
    X_A = torch.FloatTensor(X_A.todense()).to(device)
    X_B = torch.FloatTensor(X_B.todense()).to(device)

    # 转换邻接矩阵为张量并归一化
    A_A = normalize_adj_tensor(torch.FloatTensor(A_A.todense())).to(device)
    A_B = normalize_adj_tensor(torch.FloatTensor(A_B.todense())).to(device)

    # 转换 idx_test 为 torch.Tensor
    idx_test = torch.LongTensor(idx_test).to(device)

    # 定义并初始化模型
    if args.model_type == 'GCN':
        # 对于 GCN 模型
        hidden_dim = args.hidden_dim  # 32
        client_out_dim = 16  # 客户端输出维度，与训练时保持一致
        server_output_dim = int(labels.max()) + 1  # 类别数，例如 Cora 数据集为 7
        client_A = GCN(nfeat=X_A.shape[1], nclass=client_out_dim, nhid=hidden_dim, device=device,
                       dropout=args.dropout).to(device)
        client_B = GCN(nfeat=X_B.shape[1], nclass=client_out_dim, nhid=hidden_dim, device=device,
                       dropout=args.dropout).to(device)
        server_input_dim = client_out_dim * 2  # 16 + 16 = 32
        server = nn.Sequential(
            nn.Linear(server_input_dim, server_output_dim)
        ).to(device)
    elif args.model_type == 'GAT':
        # 对于 GAT 模型
        hidden_dim = args.hidden_dim  # 8
        client_out_dim = hidden_dim * args.nheads  # 8 * 2 = 16
        server_output_dim = int(labels.max()) + 1  # 类别数
        client_A = GAT(nfeat=X_A.shape[1], nhid=hidden_dim, nclass=client_out_dim, dropout=args.dropout,
                       alpha=args.alpha, nheads=args.nheads).to(device)
        client_B = GAT(nfeat=X_B.shape[1], nhid=hidden_dim, nclass=client_out_dim, dropout=args.dropout,
                       alpha=args.alpha, nheads=args.nheads).to(device)
        server_input_dim = client_out_dim * 2  # 16 + 16 = 32
        server = nn.Sequential(
            nn.Linear(server_input_dim, server_output_dim)
        ).to(device)

    # 加载训练好的模型权重
    try:
        client_A.load_state_dict(torch.load(f'{model_path}/A.pkl', map_location=device))
        client_B.load_state_dict(torch.load(f'{model_path}/B.pkl', map_location=device))
        server.load_state_dict(torch.load(f'{model_path}/server.pkl', map_location=device))
    except Exception as e:
        raise RuntimeError(f"Error loading model weights: {e}")

    # 设置模型为评估模式
    client_A.eval()
    client_B.eval()
    server.eval()

    # 对于 GAT 模型，加载 labels_B.pth
    if args.model_type == 'GAT':
        labels_B = torch.load(args.labels_B_path).to(device)

    # 定义评估函数
    def evaluate(XA, AA, XB, AB, client_A, client_B, server, labels, idx_test, XA_adv=None):
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
            client_out_A_node = client_out_A[idx_test]
            client_out_B_node = client_out_B[idx_test]
            server_input = torch.cat((client_out_A_node, client_out_B_node), dim=1)
            pred = server(server_input)
            pred = torch.log_softmax(pred, dim=1)
            predicted_labels = pred.argmax(dim=1)
            correct = (predicted_labels == labels[idx_test]).float()
            accuracy = correct.sum() / len(idx_test)
            print(f"Model Accuracy: {accuracy.item():.4f}")
            return accuracy.item()

    # 加载对抗样本
    XA_adv_numpy = np.load(adversarial_path)
    XA_adv = torch.FloatTensor(XA_adv_numpy).to(device).detach()

    # 对于 GAT 模型，使用 labels_B
    if args.model_type == 'GAT':
        labels_to_use = labels_B
    else:
        labels_to_use = labels

    # 评价模型在原始数据上的准确率
    print("Evaluating model on original data...")
    original_accuracy = evaluate(X_A, A_A, X_B, A_B, client_A, client_B, server, labels_to_use, idx_test)

    # 评价模型在对抗样本上的准确率
    print("Evaluating model on adversarial samples...")
    accuracy_after_attack = evaluate(XA_adv, A_A, X_B, A_B, client_A, client_B, server, labels_to_use, idx_test)

    print(f"Original Accuracy: {original_accuracy:.4f}")
    print(f"Accuracy after attack: {accuracy_after_attack:.4f}")

if __name__ == "__main__":
    main()
