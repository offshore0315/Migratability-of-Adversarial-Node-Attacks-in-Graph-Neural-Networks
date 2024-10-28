import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import torch
import random

np_load_old = np.load
np.aload = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)


class EarlyStop_loss:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model, file):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, file)
        elif np.isnan(score):
            print('Loss is Nan')
            self.early_stop = True
        elif score >= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, file)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model, file):
        '''Saves model when validation loss decreases.'''
        torch.save(model.state_dict(), file + '_checkpoint.pt', _use_new_zipfile_serialization=False)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# --------------------- Load data ----------------------

def load_npz(file_name):
    """Load a SparseGraph from a Numpy binary file."""
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.aload(file_name) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                   shape=loader['adj_shape'])

        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                        shape=loader['attr_shape'])
        else:
            attr_matrix = None

        labels = loader.get('labels')

    return adj_matrix, attr_matrix, labels


def largest_connected_components(adj, n_components=1):
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]
    nodes_to_keep = [idx for (idx, component) in enumerate(component_indices) if component in components_to_keep]
    print(f"Selecting {n_components} largest connected components")
    return nodes_to_keep


# ------------------------ Normalize -----------------------
# D^(-0.5) * A * D^(-0.5)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def normalize_tensor(sp_adj_tensor, edges=None, sub_graph_nodes=None, sp_degree=None):
    # 获取边的索引和权重
    edge_index = sp_adj_tensor.coalesce().indices()
    edge_weight = sp_adj_tensor.coalesce().values()
    shape = sp_adj_tensor.shape
    num_nodes = sp_adj_tensor.size(0)

    # 行和列索引
    row, col = edge_index

    # 如果提供了自定义的度数，使用它，否则根据邻接矩阵计算度数
    if sp_degree is None:
        deg = torch.sparse.sum(sp_adj_tensor, 1).to_dense().flatten()
    else:
        deg = sp_degree
        for i in range(len(edges)):
            idx = sub_graph_nodes[0, i]
            deg[idx] = deg[idx] + edges[i]
        last_deg = torch.sparse.sum(sp_adj_tensor[-1]).unsqueeze(0).data
        deg = torch.cat((deg, last_deg))

    # 处理孤立节点的度数为 0 的情况，给它们添加自环
    deg[deg == 0] = 1  # 避免度数为 0

    # 重新计算度数的平方根倒数
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    # 处理重复边和边权重匹配问题
    if len(row) != len(edge_weight):
        print(f"检测到不匹配：row size {row.size(0)}，edge_weight size {edge_weight.size(0)}。正在截取数据...")

        # 截取较长的部分，使两者大小匹配
        min_size = min(row.size(0), edge_weight.size(0))
        row = row[:min_size]
        col = col[:min_size]
        edge_weight = edge_weight[:min_size]

    # 再次检查 row 和 col 的大小是否与 edge_weight 匹配
    if row.size(0) != edge_weight.size(0):
        raise ValueError(f"归一化前大小仍不匹配：row size {row.size(0)}，edge_weight size {edge_weight.size(0)}")

    # 归一化操作
    min_size = row.size(0)
    values = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    nor_adj_tensor = torch.sparse_coo_tensor(edge_index[:, :min_size], values, shape)

    return nor_adj_tensor


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


# --------------------------------- Sub-graph ------------------------

def k_order_nei(adj, k, target):
    for i in range(k):
        if i == 0:
            one_order_nei = adj[target].nonzero()[1]
            sub_graph_nodes = one_order_nei
        else:
            sub_graph_nodes = np.unique(adj[sub_graph_nodes].nonzero()[1])

    sub_tar = np.where(sub_graph_nodes == target)[0]
    sub_idx = np.where(np.in1d(sub_graph_nodes, one_order_nei))[0]
    return one_order_nei, sub_graph_nodes, sub_tar, sub_idx


def sub_graph_tensor(two_order_nei, feat, adj, normadj, device):
    sub_feat = feat[two_order_nei]
    sub_adj = adj[two_order_nei][:, two_order_nei]
    sub_nor_adj = normadj[two_order_nei][:, two_order_nei]
    sub_adj_tensor = sparse_mx_to_torch_sparse_tensor(sub_adj).to(device)
    sub_nor_adj_tensor = sparse_mx_to_torch_sparse_tensor(sub_nor_adj).to(device)
    return sub_feat, sub_adj_tensor, sub_nor_adj_tensor


# -------------------------------------- After Attack ----------------------------------

def gen_new_adj_tensor(adj_tensor, edges, sub_graph_nodes, device):
    n = adj_tensor.shape[0]
    edge_idx = sub_graph_nodes
    sub_mask_shape = edge_idx.shape[1]
    extend_i0 = torch.cat((n * torch.ones(sub_mask_shape).unsqueeze(0).long(), edge_idx), 0)
    extend_i1 = torch.cat((edge_idx, n * torch.ones(sub_mask_shape).unsqueeze(0).long()), 0)
    extend_i = torch.cat((extend_i0, extend_i1, torch.LongTensor([[n], [n]])), 1).to(device)

    add_one = torch.ones(1).to(device)
    extend_v = torch.cat((edges, edges, add_one), 0)

    i = adj_tensor._indices()
    v = adj_tensor._values()

    new_i = torch.cat([i, extend_i], 1)
    new_v = torch.cat([v, extend_v], 0)
    new_adj_tensor = torch.sparse_coo_tensor(new_i, new_v, torch.Size([n + 1, n + 1]))
    return new_adj_tensor


def worst_case_class(logp, labels_np):
    logits_np = logp.cpu().numpy()
    max_indx = logits_np.argmax(1)
    for i, indx in enumerate(max_indx):
        logits_np[i][indx] = np.nan
        logits_np[i][labels_np[i]] = np.nan
    second_max_indx = np.nanargmax(logits_np, axis=1)

    return second_max_indx


def gen_new_adj_topo_tensor(adj_topo_tensor, edges, sub_graph_nodes, device):
    # tensor
    n = adj_topo_tensor.shape[0]
    new_edge = torch.zeros((1,n)).to(device)
    new_edge[0, sub_graph_nodes] = edges
    new_adj_topo_tensor = torch.cat((adj_topo_tensor, new_edge),dim=0)
    add_one = torch.ones((1,1)).to(device)
    new_inj_edge = torch.cat((new_edge, add_one), dim=1)
    new_adj_topo_tensor = torch.cat((new_adj_topo_tensor, new_inj_edge.reshape(n+1,1)),dim=1)
    return new_adj_topo_tensor