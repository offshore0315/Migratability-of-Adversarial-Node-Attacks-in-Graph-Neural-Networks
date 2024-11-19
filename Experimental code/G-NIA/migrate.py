import numpy as np
import os
import sys
import torch.nn.functional as F

# 修正 numpy 加载行为
np_load_old = np.load

def np_aload(*args, **k):
    if 'allow_pickle' not in k:
        k['allow_pickle'] = True
    return np_load_old(*args, **k)

np.aload = np_aload

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# 导入自定义模块
sys.path.append('..')
from utils import *
from surrogate_model.gcn import GCN
from surrogate_model.gat import GAT, LayerType

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

def main():
    # 配置参数
    gpu_id = "0"

    migrate_dataset_name = "cora"  # 源数据集名称
    dataset = 'cora'  # 目标数据集名称
    migrate_model_name = "gat"  # 源
    surro_type = 'gcn'  # 目标
    victim_type = surro_type  # 受害者模型类型
    connect = True

    # 设置 GAT 的参数
    GAT_para = {
        'citeseer': (64, 2),
        'cora': (64, 8),
        'pubmed': (64, 8)  # 修改为与检查点一致的参数 nhead=2
    }
    nhid, nhead = GAT_para.get(dataset, (8, 8))  # 为当前数据集获取 nhid 和 nhead
    print(f"Dataset: {dataset}, nhid: {nhid}, nhead: {nhead}")

    # 定义保存文件路径
    surro_save_file = 'checkpoint/surrogate_model/' + dataset + '_' + surro_type
    victim_save_file = 'checkpoint/surrogate_model/' + dataset + '_' + victim_type

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    adj, features, labels_np = load_npz(f'datasets/{dataset}.npz')
    n = adj.shape[0]
    nc = labels_np.max() + 1
    nfeat = features.shape[1]
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) + sp.eye(n)
    adj[adj > 1] = 1
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
    labels = torch.LongTensor(labels_np).to(device)

    split = np.aload('datasets/' + dataset + '_split.npy').item()
    train_mask_old = split['train']
    val_mask_old = split['val']
    test_mask_old = split['test']

    # 更新索引
    train_mask = np.array([old_to_new[idx] for idx in train_mask_old if idx in old_to_new])
    val_mask = np.array([old_to_new[idx] for idx in val_mask_old if idx in old_to_new])
    test_mask = np.array([old_to_new[idx] for idx in test_mask_old if idx in old_to_new])

    # 加载模型
    print("Loading models...")
    # Surrogate model
    if surro_type == 'gcn':
        surro_net = GCN(nfeat, 64, nc, 0.5).float().to(device)
    elif surro_type == 'gat':
        surro_net = GAT(
            num_of_layers=2,
            num_heads_per_layer=[nhead, 1],
            num_features_per_layer=[nfeat, nhid, nc],
            add_skip_connection=False,
            bias=True,
            dropout=0.6,
            layer_type=LayerType.IMP2,
            log_attention_weights=False
        ).to(device)

    # 加载模型参数
    try:
        surro_net.load_state_dict(torch.load(surro_save_file + '_checkpoint.pt', map_location=device))
    except Exception as e:
        print(f"Error loading surrogate model: {e}")
        sys.exit(1)

    # Victim model
    if victim_type == 'gcn':
        victim_net = GCN(nfeat, 64, nc, 0.5).float().to(device)
    elif victim_type == 'gat':
        victim_net = GAT(
            num_of_layers=2,
            num_heads_per_layer=[nhead, 1],
            num_features_per_layer=[nfeat, nhid, nc],
            add_skip_connection=False,
            bias=True,
            dropout=0.6,
            layer_type=LayerType.IMP2,
            log_attention_weights=False
        ).to(device)

    # 加载模型参数
    try:
        victim_net.load_state_dict(torch.load(victim_save_file + '_checkpoint.pt', map_location=device))
    except Exception as e:
        print(f"Error loading victim model: {e}")
        sys.exit(1)

    # 设置模型为评估模式
    victim_net.eval()
    surro_net.eval()
    for p in victim_net.parameters():
        p.requires_grad = False
    for p in surro_net.parameters():
        p.requires_grad = False

    # 评估模型在原始数据上的准确率
    print("Evaluating model on original data...")
    if victim_type == 'gat':
        adj_topo_tensor = torch.tensor(adj.toarray(), dtype=torch.float, device=device)
        graph_data = (feat, adj_topo_tensor)
        logits = victim_net(graph_data)[0]
    else:
        logits = victim_net(feat, nor_adj_tensor)
    logp = F.log_softmax(logits, dim=1)
    acc = accuracy(logp, labels)
    print('Original Accuracy:')
    print('Overall Acc:', acc)
    print('Train Acc:', accuracy(logp[train_mask], labels[train_mask]))
    print('Valid Acc:', accuracy(logp[val_mask], labels[val_mask]))
    print('Test Acc:', accuracy(logp[test_mask], labels[test_mask]))

    # 加载对抗样本
    print("Loading adversarial examples...")
    adv_examples_dir = 'adversarial_examples/'
    adv_file_name = os.path.join(adv_examples_dir, f'{migrate_dataset_name}_{migrate_model_name}_to_{dataset}.npy')

    if not os.path.exists(adv_file_name):
        print(f"No adversarial examples file found at {adv_file_name}. Exiting.")
        return

    adv_examples = np.load(adv_file_name, allow_pickle=True)
    print(f"Type of adv_examples: {type(adv_examples)}")
    if isinstance(adv_examples, np.ndarray) and adv_examples.size == 1:
        adv_examples = adv_examples.item()
        print("Loaded adv_examples as a single item.")
    elif isinstance(adv_examples, dict):
        print("adv_examples is already a dict.")
    else:
        print("adv_examples is in unexpected format. Exiting.")
        return

    # 检查 adv_examples 的结构
    print(f"Keys in adv_examples: {list(adv_examples.keys())}")

    injected_features = []
    injected_edges = []
    num_nodes = feat.shape[0]
    next_node_idx = num_nodes

    for dset in ['train', 'val', 'test']:
        if dset not in adv_examples:
            print(f"No adversarial examples found for dataset '{dset}'. Skipping.")
            continue
        dset_adv = adv_examples[dset]
        node_indices = dset_adv['node_idx']
        inj_feats = dset_adv['inj_feat']
        disc_scores = dset_adv['disc_score']
        masked_score_idxs = dset_adv['masked_score_idx']

        print(f"Processing {len(node_indices)} adversarial examples for dataset '{dset}'.")

        for i in range(len(node_indices)):
            inj_feat = torch.tensor(inj_feats[i]).float()
            # 调整 inj_feat 的形状以匹配目标数据集的特征维度
            if inj_feat.shape[1] != nfeat:
                # 进行特征维度的调整，例如截断或填充
                if inj_feat.shape[1] > nfeat:
                    inj_feat = inj_feat[:, :nfeat]
                else:
                    padding = torch.zeros((inj_feat.shape[0], nfeat - inj_feat.shape[1]))
                    inj_feat = torch.cat((inj_feat, padding), dim=1)
            injected_features.append(inj_feat)

            disc_score = torch.tensor(disc_scores[i])
            masked_score_idx = torch.tensor(masked_score_idxs[i])

            # 根据 disc_score 和 masked_score_idx 计算连接的边
            edge_mask = disc_score.squeeze() > 0.5  # 根据阈值确定连接的边
            if edge_mask.dim() == 0:
                edge_mask = edge_mask.unsqueeze(0)
            try:
                connected_node_indices = masked_score_idx.flatten()[edge_mask].detach().cpu().numpy()
            except IndexError as e:
                print(f"IndexError for adversarial example {i} in dataset '{dset}': {e}")
                continue  # 跳过当前对抗样本

            if connected_node_indices.size == 0:
                print(f"No edges to inject for adversarial example {i} in dataset '{dset}'.")
                continue  # 跳过当前对抗样本

            injected_node_idx = next_node_idx
            for idx in connected_node_indices:
                injected_edges.append((injected_node_idx, idx))
                injected_edges.append((idx, injected_node_idx))
            next_node_idx += 1

    # 构建新的特征矩阵和邻接矩阵
    print("Constructing attacked graph...")
    if len(injected_features) == 0:
        print("No adversarial examples found.")
        return

    injected_features_tensor = torch.cat(injected_features, dim=0)
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
    print("Evaluating model on attacked data...")
    if victim_type == 'gat':
        adj_topo_tensor = torch.tensor(new_adj.toarray(), dtype=torch.float, device=device)
        graph_data = (new_feat, adj_topo_tensor)
        logits = victim_net(graph_data)[0]
    else:
        logits = victim_net(new_feat, new_nor_adj_tensor)

    logp = F.log_softmax(logits, dim=1)

    orig_node_mask = torch.arange(new_num_nodes, device=device) < num_nodes
    acc = accuracy(logp[orig_node_mask], labels)

    print('Accuracy after attack:')
    print('Overall Acc:', acc)
    print('Train Acc:', accuracy(logp[train_mask], labels[train_mask]))
    print('Valid Acc:', accuracy(logp[val_mask], labels[val_mask]))
    print('Test Acc:', accuracy(logp[test_mask], labels[test_mask]))

if __name__ == '__main__':
    main()
