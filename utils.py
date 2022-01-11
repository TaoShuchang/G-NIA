import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from scipy.sparse.csgraph import connected_components
import torch
import torch.nn.functional as F
import random
np_load_old = np.load
np.aload = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


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
            self.save_checkpoint(model,file)
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
            self.save_checkpoint(model,file)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model, file):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), file+'_checkpoint.pt',_use_new_zipfile_serialization=False)  

 
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True      



# --------------------- Load data ----------------------

def load_npz(file_name):
    """Load a SparseGraph from a Numpy binary file.
    Parameters
    ----------
    file_name : str
        Name of the file to load.
    Returns
    -------
    sparse_graph : gust.SparseGraph
        Graph in sparse matrix format.
    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.aload(file_name) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                              loader['adj_indptr']), shape=loader['adj_shape'])

        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                   loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            attr_matrix = None

        labels = loader.get('labels')

    return adj_matrix, attr_matrix, labels

def largest_connected_components(adj, n_components=1):
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep

    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


# ------------------------ Normalize -----------------------
# D^(-0.5) * A * D^(-0.5)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    return mx

def normalize_tensor(sp_adj_tensor,edges=None, sub_graph_nodes=None,sp_degree=None):
    edge_index = sp_adj_tensor.coalesce().indices()
    edge_weight = sp_adj_tensor.coalesce().values()
    shape = sp_adj_tensor.shape
    num_nodes= sp_adj_tensor.size(0)

    row, col = edge_index
    if sp_degree is None:
        # print('None')
        deg = torch.sparse.sum(sp_adj_tensor,1).to_dense().flatten()
    else:
        # print('sp')
        deg = sp_degree
        for i in range(len(edges)):
            idx = sub_graph_nodes[0,i]
            deg[idx] = deg[idx] + edges[i]
        last_deg = torch.sparse.sum(sp_adj_tensor[-1]).unsqueeze(0).data
        deg = torch.cat((deg,last_deg))
        
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    values = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    # print('nor adj values:', values.shape)
    # print('index:',edge_index.shape)
    # print('shape:', shape)
    nor_adj_tensor = torch.sparse.FloatTensor(edge_index, values, shape)
    del edge_index, edge_weight, values, deg_inv_sqrt
    return nor_adj_tensor

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# --------------------------------- Sub-graph ------------------------ 
    
def k_order_nei(adj, k, target):
    for i in range(k):
        if i == 0:
            one_order_nei = adj[target].nonzero()[1]
            sub_graph_nodes = one_order_nei
        else:
            sub_graph_nodes = np.unique(adj[sub_graph_nodes].nonzero()[1])
        
    sub_tar = np.where(sub_graph_nodes==target)[0]
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
    # sparse tensor
    n = adj_tensor.shape[0]
    edge_idx = sub_graph_nodes
    sub_mask_shape = edge_idx.shape[1]
    extend_i0 = torch.cat((n*torch.ones(sub_mask_shape).unsqueeze(0).long(), edge_idx), 0)
    extend_i1 = torch.cat((edge_idx, n*torch.ones(sub_mask_shape).unsqueeze(0).long()), 0)
    extend_i = torch.cat((extend_i0, extend_i1,  torch.LongTensor([[n],[n]])), 1).to(device)
    
    add_one = torch.ones(1).to(device)
    extend_v = torch.cat((edges, edges, add_one),0)

    i = adj_tensor._indices()
    v = adj_tensor._values()
        
    new_i = torch.cat([i, extend_i], 1)
    new_v = torch.cat([v, extend_v], 0)
    new_adj_tensor = torch.sparse.FloatTensor(new_i, new_v, torch.Size([n+1,n+1]))
    return new_adj_tensor

def block_spmm(ori_adj_tensor, inj_adj_tensor_row, inj_adj_tensor_col, inj_adj_tensor_one, ori_feat, inj_feat, W):
    if inj_feat.dim() == 1:
        inj_feat = inj_feat.unsqueeze(0)
    ori_xw = torch.mm(ori_feat, W)
    inj_xw = torch.mm(inj_feat, W)
    ori_adj_ori_xw = torch.sparse.mm(ori_adj_tensor, ori_xw)
    injrow_adj_ori_xw = torch.sparse.mm(inj_adj_tensor_row, ori_xw)
    injcol_adj_inj_xw = torch.sparse.mm(inj_adj_tensor_col, inj_xw)
    inj_adj_inj_xw = torch.sparse.mm(inj_adj_tensor_one, inj_xw)
    
    ori_emb = ori_adj_ori_xw + injcol_adj_inj_xw
    inj_emb = injrow_adj_ori_xw + inj_adj_inj_xw
    return ori_emb, inj_emb

def approximate_evaluate_res(degree, ori_adj_tensor, ori_feat, edges, edge_idx, inj_feat, W1, W2, budget, device):
    # inject adj tensor
    n = ori_adj_tensor.shape[0]
    zeros = torch.zeros(edge_idx.shape).long()
    extend_i_row = torch.cat((zeros, edge_idx), 0).to(device)
    extend_i_col = torch.cat((edge_idx, zeros), 0).to(device)

    r_inv = np.power(budget + 1, -0.5)
    edge_idx_0 = edge_idx[0]
    sub_d = degree[edge_idx_0] + 1
    sub_d_inv = torch.pow(sub_d, -0.5)
    extend_v = sub_d_inv * edges * r_inv
    add_one = torch.FloatTensor([r_inv*r_inv])

    inj_adj_tensor_row = torch.sparse.FloatTensor(extend_i_row, extend_v, torch.Size([1, n]))
    inj_adj_tensor_col = torch.sparse.FloatTensor(extend_i_col, extend_v, torch.Size([n, 1]))
    inj_adj_tensor_one = torch.sparse.FloatTensor(torch.LongTensor([[0],[0]]), add_one, torch.Size([1, 1])).to(device)
    
    ori_emb1, inj_emb1 = block_spmm(ori_adj_tensor, inj_adj_tensor_row, inj_adj_tensor_col, inj_adj_tensor_one, ori_feat, inj_feat, W1)
    ori_emb1 = F.relu(ori_emb1)
    inj_emb1 = F.relu(inj_emb1)
    ori_emb2, inj_emb2 = block_spmm(ori_adj_tensor, inj_adj_tensor_row, inj_adj_tensor_col, inj_adj_tensor_one, ori_emb1, inj_emb1, W2)

    approimate_emb = torch.cat((ori_emb2, inj_emb2))
    return approimate_emb


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

def gen_new_edge_idx(adj_edge_index, disc_score, masked_score_idx, device):
    inj_node = adj_edge_index.max() + 1
    inj_sub_idx = torch.where(disc_score>=0.9)[0]
    inj_edge_idx = masked_score_idx[0,inj_sub_idx].unsqueeze(0)
    inj_idx = inj_node.repeat(inj_edge_idx.shape)
    pos_inj_edges = torch.cat((inj_idx, inj_idx),dim=0).to(device)
    rev_inj_edges = torch.cat((inj_idx, inj_idx),dim=0).to(device)
    new_edge_idx = torch.cat((adj_edge_index, pos_inj_edges, rev_inj_edges),dim=1)
    return new_edge_idx


# ----------------------------- ACC --------------------------


def worst_case_class(logp, labels_np):
    logits_np = logp.cpu().numpy()
    max_indx = logits_np.argmax(1)
    for i, indx in enumerate(max_indx):
        logits_np[i][indx] = np.nan
        logits_np[i][labels_np[i]] = np.nan
    second_max_indx = np.nanargmax(logits_np, axis=1)

    return second_max_indx


        
def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def train_val_test_split_tabular(arrays, train_size=0.1, val_size=0.1, test_size=0.8, stratify=None, random_state=123):
    idx = arrays
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=random_state,
                                                   train_size=(train_size + val_size),
                                                   test_size=test_size,
                                                   stratify=stratify)
    if stratify is not None:
        stratify = stratify[idx_train_and_val]
    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=random_state,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)

    return idx_train, idx_val, idx_test


# ------------------------------ Multi-targets ------------------------

def obtain_multi_targets(dataset, tar_num, adj, seed=123):
    deg = np.array(adj.sum(1)).squeeze()
    deg_sort = deg.argsort()
    tmp_idx = np.where(deg[deg_sort]>=tar_num)
    idx = deg_sort[tmp_idx]         # idx contains nodes whose degree are larger than tar_num
    
    candidate = np.arange(adj.shape[0])
    real_targets_all = []
    for i in idx:
        one_order_nei = adj[i].nonzero()[1]
        cand_one_order_nei = np.intersect1d(candidate,one_order_nei)
        if len(cand_one_order_nei) >= tar_num:
            real_targets = np.random.choice(cand_one_order_nei, tar_num, replace=False)
            real_targets_all.append(real_targets)
            candidate = np.setdiff1d(candidate, real_targets)
    real_targets_arr = np.array(real_targets_all)
    
    mask = np.arange(len(real_targets_arr))
    train_mask, val_mask, test_mask = train_val_test_split_tabular(mask, train_size=0.64, val_size=0.16, test_size=0.2, random_state=seed)
    split={}
    split['train'] = train_mask
    split['val'] = val_mask
    split['test'] = test_mask
    np.save('datasets/multargets_'+dataset + '_tarnum' + str(tar_num) + '.npy', real_targets_arr)
    np.save('datasets/multargets_'+dataset+ '_tarnum' + str(tar_num) + '_split.npy',split)
    return 
    