import os
import argparse

import torch
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import scipy.sparse as sp

from surrogate_model.gcn import GCN
from surrogate_model.gat import GAT, LayerType
from utils import *

def setup_seed(seed):
    """
    固定随机种子，确保实验的可重复性。
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # 启用CUDNN基准测试以提高性能

def accuracy(output, labels):
    """
    计算模型的准确率。
    """
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def evaluate_model(model_path, device, model_type, feat, nor_adj_tensor, labels, train_mask, val_mask, test_mask, victim_type, adj):
    """
    加载保存的模型并评估其在训练集、验证集和测试集上的准确率。

    参数:
        model_path (str): 模型文件的路径（.pt 文件）。
        device (torch.device): 设备（CPU 或 GPU）。
        model_type (str): 模型类型（'gcn' 或 'gat'）。
        feat (torch.Tensor): 特征矩阵。
        nor_adj_tensor (torch.Tensor): 归一化后的邻接矩阵。
        labels (torch.Tensor): 标签向量。
        train_mask (torch.LongTensor): 训练集掩码。
        val_mask (torch.LongTensor): 验证集掩码。
        test_mask (torch.LongTensor): 测试集掩码。
        victim_type (str): 评估模型的类型（如 'gcn'）。
        adj (scipy.sparse matrix): 原始邻接矩阵。
    """
    # 创建模型实例
    nc = labels.max().item() + 1
    nfeat = feat.shape[1]

    if model_type == 'gcn':
        model = GCN(nfeat, 64, nc, dropout=0.5).to(device)
    elif model_type == 'gat':
        model = GAT(
            num_of_layers=2,
            num_heads_per_layer=[4, 1],  # 如果需要，可以减少头数以提高性能
            num_features_per_layer=[nfeat, 64, nc],
            add_skip_connection=False,
            bias=True,
            dropout=0.6,
            layer_type=LayerType.IMP2,
            log_attention_weights=False
        ).to(device)
    else:
        raise ValueError("Unsupported model type. Choose 'gcn' or 'gat'.")

    # 加载模型参数
    if not os.path.exists(model_path):
        print(f"模型文件 {model_path} 不存在。")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # 评估模型
    if victim_type == 'gat':
        # 如果 victim_type 为 'gat'，需要构建邻接矩阵的dense形式
        adj_topo_tensor = torch.tensor(adj.toarray(), dtype=torch.float, device=device)
        graph_data = (feat, adj_topo_tensor)
        logits = model(graph_data)[0]
    else:
        logits = model(feat, nor_adj_tensor)

    logp = F.log_softmax(logits, dim=1)
    acc = accuracy(logp, labels)
    print('Loaded Model Accuracy:')
    print(f'Overall Acc: {acc:.4f}')
    print(f'Train Acc: {accuracy(logp[train_mask], labels[train_mask]):.4f}')
    print(f'Valid Acc: {accuracy(logp[val_mask], labels[val_mask]):.4f}')
    print(f'Test Acc: {accuracy(logp[test_mask], labels[test_mask]):.4f}')

def train_model(args):
    """
    训练指定类型的模型（GCN 或 GAT），并保存和评估模型。
    """
    # 固定随机种子
    setup_seed(args.seed)

    # 配置参数
    dataset = args.dataset
    model_type = args.model_type.lower()
    victim_type = model_type  # 假设 victim_type 与 model_type 相同
    connect = args.connect
    discrete = args.discrete
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    surro_type = model_type  # 假设 surrogate_type 与 model_type 相同

    # 定义保存文件路径
    victim_save_file = os.path.join('checkpoint', 'surrogate_model', f'{dataset}_{model_type}')
    os.makedirs(os.path.dirname(victim_save_file), exist_ok=True)

    # 加载数据
    adj, features, labels_np = load_npz(f'datasets/{dataset}.npz')
    n = adj.shape[0]
    nc = labels_np.max() + 1
    nfeat = features.shape[1]
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) + sp.eye(n)
    adj[adj > 1] = 1  # 保证邻接矩阵为0或1

    if connect:
        lcc = largest_connected_components(adj)
        adj = adj[lcc][:, lcc]
        features = features[lcc]
        labels_np = labels_np[lcc]
        n = adj.shape[0]
        print('节点数量:', n)
        # 创建旧索引到新索引的映射
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(lcc)}
    else:
        old_to_new = {idx: idx for idx in range(n)}

    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    nor_adj_tensor = normalize_tensor(adj_tensor)

    feat = torch.from_numpy(features.todense().astype('double')).float().to(device)
    labels = torch.LongTensor(labels_np).to(device)

    # 更新掩码
    split = np.aload(f'datasets/{dataset}_split.npy').item()
    train_mask_old = split['train']
    val_mask_old = split['val']
    test_mask_old = split['test']

    train_mask = np.array([old_to_new[idx] for idx in train_mask_old if idx in old_to_new])
    val_mask = np.array([old_to_new[idx] for idx in val_mask_old if idx in old_to_new])
    test_mask = np.array([old_to_new[idx] for idx in test_mask_old if idx in old_to_new])

    # 将mask转换为torch tensor
    train_mask = torch.LongTensor(train_mask).to(device)
    val_mask = torch.LongTensor(val_mask).to(device)
    test_mask = torch.LongTensor(test_mask).to(device)

    # 创建模型
    if model_type == 'gcn':
        model = GCN(nfeat, 64, nc, dropout=0.5).to(device)
    elif model_type == 'gat':
        model = GAT(
            num_of_layers=2,
            num_heads_per_layer=[args.nhead, 1],  # 可以根据需要减少头数
            num_features_per_layer=[nfeat, 64, nc],
            add_skip_connection=False,
            bias=True,
            dropout=0.6,
            layer_type=LayerType.IMP2,
            log_attention_weights=False
        ).to(device)
    else:
        raise ValueError("Unsupported model type. Choose 'gcn' or 'gat'.")

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    # 预先计算GAT所需的密集邻接矩阵
    if model_type == 'gat':
        adj_dense = adj.toarray()
        adj_topo_tensor = torch.tensor(adj_dense, dtype=torch.float, device=device)
        graph_data = (feat, adj_topo_tensor)
    else:
        graph_data = None  # GCN不需要

    # 训练模型
    model.train()
    for epoch in range(args.nepochs):
        optimizer.zero_grad()
        if model_type == 'gat':
            logits = model(graph_data)[0]
        else:
            logits = model(feat, nor_adj_tensor)
        loss = loss_fn(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        # 评估模型
        model.eval()
        with torch.no_grad():
            if model_type == 'gat':
                logits = model(graph_data)[0]
            else:
                logits = model(feat, nor_adj_tensor)
            train_acc = accuracy(logits[train_mask], labels[train_mask])
            val_acc = accuracy(logits[val_mask], labels[val_mask])
            test_acc = accuracy(logits[test_mask], labels[test_mask])
        model.train()
        print(f'Epoch {epoch+1}/{args.nepochs}, Loss: {loss.item():.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

    # 保存模型
    model_path = f'{victim_save_file}_checkpoint.pt'
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到 {model_path}")

    # 评估保存的模型
    evaluate_model(
        model_path=model_path,
        device=device,
        model_type=model_type,
        feat=feat,
        nor_adj_tensor=nor_adj_tensor,
        labels=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        victim_type=victim_type,
        adj=adj  # 传递原始邻接矩阵用于评估
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GCN or GAT model on a dataset.')

    # 一般配置
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--gpu', type=str, default="0", help='GPU ID')
    parser.add_argument('--dataset', default='pubmed', help='Dataset to use')
    parser.add_argument('--model_type', default='gcn', choices=['gcn', 'gat'], help='Model type to train')
    parser.add_argument('--connect', action='store_true', help='Whether to extract the largest connected component')
    parser.add_argument('--discrete', type=bool, default=True, help='Discrete attribute flag')

    # 模型特定的超参数
    parser.add_argument('--nhead', type=int, default=2, help='Number of attention heads for GAT')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--nepochs', type=int, default=200, help='Number of training epochs')

    args = parser.parse_args()

    # 设置CUDA设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 开始训练
    train_model(args)

    # pubmed gcn 0.7688
