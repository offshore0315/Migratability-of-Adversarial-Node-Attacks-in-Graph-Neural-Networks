import sys
import os
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data

from gnia import GNIA
from utils import *
from surrogate_model.gcn import GCN
from surrogate_model.gat import GAT, LayerType

from torch.cuda.amp import autocast, GradScaler  # 导入混合精度训练模块（可选）


# 固定随机种子，确保实验的可重复性
def setup_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # 启用CUDNN基准测试以提高性能


setup_seed(123)


def accuracy(output, labels):
    """
    计算模型的准确率。
    """
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def evaluate_model(model_path, device, model_type, feat, nor_adj_tensor, labels, train_mask, val_mask, test_mask,
                   victim_type, adj):
    # 创建模型实例
    nc = labels.max().item() + 1
    nfeat = feat.shape[1]

    if model_type == 'gcn':
        model = GCN(nfeat, 64, nc, dropout=0.5).to(device)
    elif model_type == 'gat':
        model = GAT(
            num_of_layers=2,
            num_heads_per_layer=[2, 1],  # 确保与检查点一致的nhead=2
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


def main(opts):
    # Hyperparameters
    gpu_id = opts['gpu']
    seed = opts['seed']
    surro_type = opts['surro_type']
    victim_type = opts['victim_type']
    dataset = opts['dataset']
    connect = opts['connect']
    multi = opts['multiedge']
    discrete = opts['discrete']
    suffix = opts['suffix']
    attr_tau = float(opts['attrtau']) if opts['attrtau'] is not None else 0.75  # 默认值0.75
    edge_tau = float(opts['edgetau']) if opts['edgetau'] is not None else 0.75  # 默认值0.75
    lr = opts['lr']
    patience = opts['patience']
    best_score = opts['best_score']
    counter = opts['counter']
    nepochs = opts['nepochs']
    st_epoch = opts['st_epoch']
    epsilon_start = opts['epsst']
    epsilon_end = 0
    epsilon_decay = opts['epsdec']
    total_steps = 500
    batch_size = opts['batchsize']
    nhid = opts['nhid']
    nhead = opts['nhead']

    # 定义保存文件路径
    surro_save_file = os.path.join('checkpoint', 'surrogate_model', f'{dataset}_{surro_type}')
    victim_save_file = os.path.join('checkpoint', 'surrogate_model', f'{dataset}_{victim_type}')
    ckpt_save_dirs = os.path.join('checkpoint', f'{surro_type}_gnia')
    output_save_dirs = os.path.join('output', f'{surro_type}_gnia')
    model_save_file = os.path.join(ckpt_save_dirs, f'{dataset}_{suffix}')
    os.makedirs(ckpt_save_dirs, exist_ok=True)
    os.makedirs(output_save_dirs, exist_ok=True)

    # 创建 adversarial_examples 目录
    adv_examples_dir = 'adversarial_examples'
    os.makedirs(adv_examples_dir, exist_ok=True)

    # 初始化用于存储对抗样本的字典
    adv_examples = {
        'train': {'node_idx': [], 'inj_feat': [], 'disc_score': [], 'masked_score_idx': []},
        'val': {'node_idx': [], 'inj_feat': [], 'disc_score': [], 'masked_score_idx': []},
        'test': {'node_idx': [], 'inj_feat': [], 'disc_score': [], 'masked_score_idx': []}
    }

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # Preprocessing data
    adj, features, labels_np = load_npz(os.path.join('datasets', f'{dataset}.npz'))
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
        old_to_new = {idx: idx for idx in range(n)}  # 确保 old_to_new 被定义

    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    nor_adj_tensor = normalize_tensor(adj_tensor)

    feat = torch.from_numpy(features.todense().astype('double')).float().to(device)
    feat_max = feat.max(0).values
    feat_min = feat.min(0).values
    labels = torch.LongTensor(labels_np).to(device)
    degree = adj.sum(1)
    deg = torch.FloatTensor(degree).flatten().to(device)
    feat_num = int(features.sum(1).mean())
    eps_threshold = [epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * steps / epsilon_decay) for steps in
                     range(total_steps)]

    split = np.load(os.path.join('datasets', f'{dataset}_split.npy'), allow_pickle=True).item()
    train_mask_old = split['train']
    val_mask_old = split['val']
    test_mask_old = split['test']

    # 更新索引
    train_mask = np.array([old_to_new[idx] for idx in train_mask_old if idx in old_to_new])
    val_mask = np.array([old_to_new[idx] for idx in val_mask_old if idx in old_to_new])
    test_mask = np.array([old_to_new[idx] for idx in test_mask_old if idx in old_to_new])

    print("Surrogate GNN Model:", surro_type)
    print("Evaluation GNN Model:", victim_type)

    # 设置 GAT 的参数
    GAT_para = {
        '12k_reddit': (4, 4),
        '10k_ogbproducts': (6, 6),
        'citeseer': (64, 8),
        'cora': (64, 8),
        'pubmed': (64, 2)  # 修改为与检查点一致的参数 nhead=2
    }
    opts['nhid'], opts['nhead'] = GAT_para.get(opts['dataset'], (8, 8))  # 添加默认值
    opts['discrete'] = False if 'k_' in opts['dataset'] else True
    print(opts)

    # 加载预训练的 Surrogate 模型
    if surro_type == 'gat':
        surro_net = GAT(
            num_of_layers=2,
            num_heads_per_layer=[opts['nhead'], 1],  # 确保与检查点一致的nhead=2
            num_features_per_layer=[nfeat, opts['nhid'], nc],
            add_skip_connection=False,
            bias=True,
            dropout=0.6,
            layer_type=LayerType.IMP2,
            log_attention_weights=False
        ).to(device)
    elif surro_type == 'gcn':
        surro_net = GCN(features.shape[1], 64, labels.max().item() + 1, 0.5).float().to(device)

    # 加载模型参数
    try:
        surro_net.load_state_dict(
            torch.load(surro_save_file + '_checkpoint.pt', weights_only=True, map_location=torch.device('cpu')))
    except AttributeError as e:
        print(f"Error loading surrogate model state_dict: {e}")
        print("请确保 surrogate_model.gat.GAT 类定义与保存检查点时使用的定义一致。")
        sys.exit(1)

    # 加载预训练的 Victim 模型
    if victim_type == 'gat':
        victim_net = GAT(
            num_of_layers=2,
            num_heads_per_layer=[opts['nhead'], 1],  # 确保与检查点一致的nhead=2
            num_features_per_layer=[nfeat, opts['nhid'], nc],
            add_skip_connection=False,
            bias=True,
            dropout=0.6,
            layer_type=LayerType.IMP2,
            log_attention_weights=False
        ).to(device)
    elif victim_type == 'gcn':
        victim_net = GCN(features.shape[1], 64, labels.max().item() + 1, 0.5).float().to(device)

    # 加载模型参数
    try:
        victim_net.load_state_dict(
            torch.load(victim_save_file + '_checkpoint.pt', weights_only=True, map_location=torch.device('cpu')))
    except AttributeError as e:
        print(f"Error loading victim model state_dict: {e}")
        print("请确保 surrogate_model.gat.GAT 类定义与保存检查点时使用的定义一致。")
        sys.exit(1)

    surro_net.eval()
    victim_net.eval()
    for p in victim_net.parameters():
        p.requires_grad = False
    for p in surro_net.parameters():
        p.requires_grad = False

    # 获取节点嵌入和权重矩阵
    if surro_type == 'gcn':
        node_emb = surro_net(feat, nor_adj_tensor)
        W1 = surro_net.gc1.weight.data.detach()
        W2 = surro_net.gc2.weight.data.detach()
    elif surro_type == 'gat':
        adj_topo_tensor = torch.tensor(adj.toarray(), dtype=torch.float, device=device)
        graph_data = (feat, adj_topo_tensor)
        node_emb = surro_net(graph_data)[0]
        # 修改层属性名称为 gat_net
        W1 = surro_net.gat_net[0].linear_proj.weight.data.detach().t()
        W2 = surro_net.gat_net[1].linear_proj.weight.data.detach().t()
    else:
        node_emb = surro_net(feat, nor_adj_tensor)
        W1 = surro_net.fcs[0].weight.data.detach()
        W2 = surro_net.fcs[1].weight.data.detach().t()
    W = torch.mm(W1, W2).t()

    # 获取受害者模型的 logits 和二次优化
    if victim_type == 'gat':
        adj_topo_tensor = torch.tensor(adj.toarray(), dtype=torch.float, device=device)
        graph_data = (feat, adj_topo_tensor)
        logits = victim_net(graph_data)[0]
    else:
        logits = victim_net(feat, nor_adj_tensor)
    sec = worst_case_class(logits, labels_np)
    logp = F.log_softmax(logits, dim=1)
    acc = accuracy(logp, labels)

    # 初始化 GNIA 攻击模型
    model = GNIA(labels, nfeat, W1, W2, discrete, device, feat_min=feat_min, feat_max=feat_max, feat_num=feat_num,
                 attr_tau=attr_tau, edge_tau=edge_tau).to(device)
    stopper = EarlyStop_loss(patience=patience)

    # 定义优化器
    if opts['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=lr)
    elif opts['optimizer'] == 'RMSprop':
        optimizer = torch.optim.RMSprop([{'params': model.parameters()}], lr=lr, weight_decay=0)
    else:
        raise ValueError('Unsupported argument for the optimizer')

    # 创建数据加载器
    x = torch.LongTensor(train_mask)
    y = labels[train_mask].to(torch.device('cpu'))
    torch_dataset = Data.TensorDataset(x, y)
    batch_loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    # 加载预训练的攻击模型（如果有）
    if st_epoch != 0:
        model.load_state_dict(torch.load(model_save_file + '_checkpoint.pt'))
        stopper.best_score = best_score
        stopper.counter = counter

    atk_suc = []
    injected_features = []
    injected_edges = []
    num_nodes = feat.shape[0]
    next_node_idx = num_nodes

    for dset in ['train', 'val', 'test']:
        # 初始化每个数据集的攻击成功列表
        atk_suc_dset = []
        current_mask = train_mask if dset == 'train' else (val_mask if dset == 'val' else test_mask)
        for batch in current_mask:
            target = np.array([batch])
            target_deg = int(sum([degree[i].item() for i in target]))
            budget = int(min(round(target_deg / 2), round(degree.mean()))) if multi else 1
            best_wrong_label = sec[target[0]]
            ori = labels_np[target].item()
            one_order_nei = adj[target].nonzero()[1]

            if surro_type == 'gat':
                one_order_nei, four_order_nei, sub_tar, sub_idx = k_order_nei(adj.toarray(), 3, target)
                tar_norm_adj = nor_adj_tensor[sub_tar.item()].to_dense()
                norm_a_target = tar_norm_adj[sub_idx].unsqueeze(1)
                sub_feat = feat[four_order_nei]
                sub_adj = adj.toarray()[four_order_nei][:, four_order_nei]
                sub_adj_tensor = torch.tensor(sub_adj, dtype=torch.float, device=device)
                inj_feat, disc_score, masked_score_idx = model(sub_tar, sub_idx, budget, sub_feat, norm_a_target,
                                                               node_emb[four_order_nei],
                                                               W[ori], W[best_wrong_label], train_flag=False,
                                                               eps=0)  # eps=0 表示不使用扰动
                new_feat = torch.cat((sub_feat, inj_feat.unsqueeze(0)), 0)
            else:
                tar_norm_adj = nor_adj_tensor[target.item()].to_dense()
                norm_a_target = tar_norm_adj[one_order_nei].unsqueeze(1)
                inj_feat, disc_score, masked_score_idx = model(target, one_order_nei, budget, feat, norm_a_target,
                                                               node_emb,
                                                               W[ori], W[best_wrong_label], train_flag=False,
                                                               eps=0)  # eps=0 表示不使用扰动
                new_feat = torch.cat((feat, inj_feat.unsqueeze(0)), 0)

            # 收集对抗样本到 adv_examples 字典中
            adv_examples[dset]['node_idx'].append(batch)
            adv_examples[dset]['inj_feat'].append(inj_feat.detach().cpu().numpy().tolist())
            adv_examples[dset]['disc_score'].append(disc_score.detach().cpu().numpy().tolist())
            adv_examples[dset]['masked_score_idx'].append(masked_score_idx.detach().cpu().numpy().tolist())

            # 收集注入的特征和边
            injected_features.append(inj_feat.detach().cpu())

            # 根据 disc_score 和 masked_score_idx 计算连接的边
            edge_mask = disc_score.squeeze() > 0.5  # 根据阈值确定连接的边
            if edge_mask.dim() == 0:
                edge_mask = edge_mask.unsqueeze(0)
            try:
                connected_node_indices = masked_score_idx.flatten()[edge_mask].detach().cpu().numpy()
            except IndexError as e:
                print(f"IndexError for adversarial example in dataset '{dset}': {e}")
                continue  # 跳过当前对抗样本

            if connected_node_indices.size == 0:
                print(f"No edges to inject for adversarial example in dataset '{dset}'.")
                continue  # 跳过当前对抗样本

            injected_node_idx = next_node_idx
            for idx in connected_node_indices:
                injected_edges.append((injected_node_idx, idx))
                injected_edges.append((idx, injected_node_idx))
            next_node_idx += 1

            # 评估模型预测
            if victim_type == 'gat':
                new_adj_tensor = gen_new_adj_topo_tensor(sub_adj_tensor, disc_score, sub_idx, device)
                new_logits = victim_net((new_feat, new_adj_tensor))[0]
                new_logp = F.log_softmax(new_logits, dim=1)
                pred_label = new_logits[sub_tar].argmax(1).item()
            else:
                new_adj_tensor = gen_new_adj_tensor(adj_tensor, disc_score, masked_score_idx, device)
                new_logits = victim_net(new_feat, normalize_tensor(new_adj_tensor))
                new_logp = F.log_softmax(new_logits, dim=1)
                pred_label = new_logits[target].argmax(1).item()

            if ori != pred_label:
                # 攻击成功
                atk_suc_dset.append(1)
                atk_suc.append(1)
            else:
                # 攻击失败
                atk_suc_dset.append(0)
                atk_suc.append(0)

        print(f'Attack success rate of {dset} set:', np.array(atk_suc_dset).mean())
        print('*' * 30)

    # 构建新的特征矩阵和邻接矩阵
    print("Constructing attacked graph...")
    if len(injected_features) == 0:
        print("No adversarial examples found.")
        return

    injected_features_tensor = torch.stack(injected_features)
    new_feat = torch.cat((feat.cpu(), injected_features_tensor), dim=0).to(device)

    adj_coo = adj.tocoo()
    rows = adj_coo.row
    cols = adj_coo.col
    data = adj_coo.data

    for (i, j) in injected_edges:
        rows = np.append(rows, i)
        cols = np.append(cols, j)
        data = np.append(data, 1)

    new_num_nodes = next_node_idx
    new_adj = sp.coo_matrix((data, (rows, cols)), shape=(new_num_nodes, new_num_nodes))
    new_adj = new_adj.maximum(new_adj.transpose())

    new_adj_tensor = sparse_mx_to_torch_sparse_tensor(new_adj).to(device)
    new_nor_adj_tensor = normalize_tensor(new_adj_tensor)

    # 评估模型在被攻击的图上的准确率
    print("Evaluating model on attacked graph...")
    if victim_type == 'gat':
        adj_topo_tensor = torch.tensor(new_adj.toarray(), dtype=torch.float, device=device)
        graph_data = (new_feat, adj_topo_tensor)
        logits = victim_net(graph_data)[0]
    else:
        logits = victim_net(new_feat, new_nor_adj_tensor)

    logp = F.log_softmax(logits, dim=1)

    orig_node_mask = torch.arange(new_num_nodes, device=device) < num_nodes
    acc = accuracy(logp[orig_node_mask], labels)

    # 保存攻击成功率
    np.save(os.path.join(output_save_dirs, f'{dataset}_{suffix}_atk_success.npy'), np.array(atk_suc))

    # 保存所有对抗样本到一个 .npy 文件中
    adv_file_name = os.path.join(adv_examples_dir, f'{dataset}_{surro_type}')
    np.save(adv_file_name, adv_examples)
    print(f'All adversarial examples have been saved to {adv_file_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNIA')

    # 一般配置
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--gpu', type=str, default="0", help='GPU ID')
    parser.add_argument('--suffix', type=str, default='_', help='suffix of the checkpoint')

    # Dataset
    parser.add_argument('--dataset', default='cora', help='dataset to use')
    parser.add_argument('--surro_type', default='gcn', help='surrogate gnn model')
    parser.add_argument('--victim_type', default='gcn', help='victim gnn model')
    parser.add_argument('--connect', action='store_true', help='largest connected component')
    parser.add_argument('--multiedge', default=False, type=bool,
                        help='budget of malicious edges connected to injected node')

    # Optimization
    parser.add_argument('--optimizer', choices=['Adam', 'RMSprop'], default='Adam', help='optimizer')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--wd', default=0., type=float, help='weight decay')
    parser.add_argument('--nepochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--patience', default=100, type=int, help='patience of early stopping')
    parser.add_argument('--batchsize', type=int, default=32, help='batchsize')

    # Hyperparameters
    parser.add_argument('--attrtau', default=None,
                        help='tau of gumbel softmax on attribute on discrete attributed graph')
    parser.add_argument('--edgetau', default=None, help='tau of gumbel softmax on edge')
    parser.add_argument('--epsdec', default=20, type=float, help='epsilon decay: coefficient of the gumbel sampling')
    parser.add_argument('--epsst', default=20, type=int, help='epsilon start: coefficient of the gumbel sampling')

    # Ignorable
    parser.add_argument('--counter', type=int, default=0, help='counter for recover training (Ignorable)')
    parser.add_argument('--best_score', type=float, default=0., help='best score for recover training (Ignorable)')
    parser.add_argument('--st_epoch', type=int, default=0, help='start epoch for recover training (Ignorable)')
    parser.add_argument('--local_rank', type=int, default=2, help='DDP local rank for parallel (Ignorable)')

    args = parser.parse_args()
    opts = args.__dict__.copy()

    # 设置 GAT 的参数
    GAT_para = {
        '12k_reddit': (4, 4),
        '10k_ogbproducts': (6, 6),
        'citeseer': (64, 8),
        'cora': (64, 8),
        'pubmed': (64, 2)  # v nbnhead=2
    }
    opts['nhid'], opts['nhead'] = GAT_para.get(opts['dataset'], (8, 8))  # 添加默认值
    opts['discrete'] = False if 'k_' in opts['dataset'] else True
    print(opts)

    main(opts)
