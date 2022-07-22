from math import floor, sqrt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC

device = 'cuda'
def schlichtkrull_std(shape, gain):
    """
    a = \text{gain} \times \frac{3}{\sqrt{\text{fan\_in} + \text{fan\_out}}}
    """
    fan_in, fan_out = shape[0], shape[1]
    return gain * 3.0 / sqrt(float(fan_in + fan_out))

def schlichtkrull_normal_(tensor, shape, gain=1.):
    """Fill the input `Tensor` with values according to the Schlichtkrull method, using a normal distribution."""
    std = schlichtkrull_std(shape, gain)
    with torch.no_grad():
        return tensor.normal_(0.0, std)

def schlichtkrull_uniform_(tensor, gain=1.):
    """Fill the input `Tensor` with values according to the Schlichtkrull method, using a uniform distribution."""
    std = schlichtkrull_std(tensor, gain)
    with torch.no_grad():
        return tensor.uniform_(-std, std)

def select_b_init(init):
    """Return functions for initialising biases"""
    init = init.lower()
    if init in ['zeros', 'zero', 0]:
        return torch.nn.init.zeros_
    elif init in ['ones', 'one', 1]:
        return torch.nn.init.ones_
    elif init == 'uniform':
        return torch.nn.init.uniform_
    elif init == 'normal':
        return torch.nn.init.normal_
    else:
        raise NotImplementedError(f'{init} initialisation has not been implemented!')

def select_w_init(init):
    """Return functions for initialising weights"""
    init = init.lower()
    if init in ['glorot-uniform', 'xavier-uniform']:
        return torch.nn.init.xavier_uniform_
    elif init in ['glorot-normal', 'xavier-normal']:
        return torch.nn.init.xavier_normal_
    elif init == 'schlichtkrull-uniform':
        return schlichtkrull_uniform_
    elif init == 'schlichtkrull-normal':
        return schlichtkrull_normal_
    elif init in ['normal', 'standard-normal']:
        return torch.nn.init.normal_
    elif init == 'uniform':
        return torch.nn.init.uniform_
    else:
        raise NotImplementedError(f'{init} initialisation has not been implemented!')

def drop_edges(triples, num_nodes, general_edo, self_loop_edo):
    """ Performs edge dropout by actually removing the triples """
    general_keep = 1.0 - general_edo
    self_loop_keep = 1.0 - self_loop_edo

    # Notes: self-loop triples were appended to the end of the list in add_inverse_and_self
    nt = triples.size(0) - num_nodes

    general_keep_ind = random.sample(range(nt), k=int(floor(general_keep * nt)))
    self_loop_keep_ind = random.sample(range(nt, nt + num_nodes), k=int(floor(self_loop_keep * num_nodes)))
    ind = general_keep_ind + self_loop_keep_ind

    return triples[ind, :]

def sum_sparse(indices, values, size, row_normalisation=True, device='cpu'):
    """
    Sum the rows or columns of a sparse matrix, and redistribute the
    results back to the non-sparse row/column entries
    Arguments are interpreted as defining sparse matrix.

    Source: https://github.com/pbloem/gated-rgcn/blob/1bde7f28af8028f468349b2d760c17d5c908b58b/kgmodels/util/util.py#L304
    """
    assert len(indices.size()) == len(values.size()) + 1
    k, r = indices.size()
    if not row_normalisation:
        # Transpose the matrix for column-wise normalisation
        indices = torch.cat([indices[:, 1:2], indices[:, 0:1]], dim=1)
        size = size[1], size[0]
    ones = torch.ones((size[1], 1), device=device)
    if device == 'cuda':
        values = torch.cuda.sparse.FloatTensor(indices.t(), values, torch.Size(size))
    else:
        values = torch.sparse.FloatTensor(indices.t(), values, torch.Size(size))
    sums = torch.spmm(values, ones)
    sums = sums[indices[:, 0], 0]
    return sums.view(k)


def generate_inverses(triples, num_rels):
    """ Generates nverse relations """
    inverse_relations = torch.cat([triples[:, 2, None], triples[:, 1, None], triples[:, 0, None]], dim=1)
    assert inverse_relations.size() == triples.size()
    return inverse_relations


def generate_self_loops(triples, num_nodes, num_rels, self_loop_keep_prob, device='cpu'):
    """ Generates self-loop triples and then applies edge dropout """

    # Create a new relation id for self loop relation.
    all = torch.arange(num_nodes, device=device)[:, None]
    id  = torch.empty(size=(num_nodes, 1), device=device, dtype=torch.long).fill_(num_rels)
    self_loops = torch.cat([all, id, all], dim=1)
    assert self_loops.size() == (num_nodes, 3)

    # Apply edge dropout
    mask = torch.bernoulli(torch.empty(size=(num_nodes,), dtype=torch.float, device=device).fill_(
        self_loop_keep_prob)).to(torch.bool)
    self_loops = self_loops[mask, :]

    return self_loops 


def add_inverse_and_self(triples, num_nodes, num_rels, device='cpu'):
    """ Adds inverse relations and self loops to a tensor of triples """

    # Swap around head and tail. Create new relation ids for inverse relations.
    inverse_relations = torch.cat([triples[:, 2, None], triples[:, 1, None] + num_rels, triples[:, 0, None]], dim=1)
    assert inverse_relations.size() == triples.size()

    # Create a new relation id for self loop relation.
    all = torch.arange(num_nodes, device=device)[:, None]
    id  = torch.empty(size=(num_nodes, 1), device=device, dtype=torch.long).fill_(2*num_rels)
    self_loops = torch.cat([all, id, all], dim=1)
    assert self_loops.size() == (num_nodes, 3)

    # Note: Self-loops are appended to the end and this makes it easier to apply different edge dropout rates.
    return torch.cat([triples, inverse_relations, self_loops], dim=0)

def stack_matrices(triples, num_nodes, num_rels, vertical_stacking=True, device='cpu'):
    """
    Computes a sparse adjacency matrix for the given graph (the adjacency matrices of all
    relations are stacked vertically).
    """
    assert triples.dtype == torch.long

    r, n = num_rels, num_nodes
    size = (r * n, n) if vertical_stacking else (n, r * n)

    fr, to = triples[:, 0], triples[:, 2]
    offset = triples[:, 1] * n
    if vertical_stacking:
        fr = offset + fr
    else:
        to = offset + to

    indices = torch.cat([fr[:, None], to[:, None]], dim=1).to(device)

    assert indices.size(0) == triples.size(0)
    assert indices[:, 0].max() < size[0], f'{indices[0, :].max()}, {size}, {r}'
    assert indices[:, 1].max() < size[1], f'{indices[1, :].max()}, {size}, {r}'

    return indices, size

def block_diag(m):
    """
    Source: https://gist.github.com/yulkang/2e4fc3061b45403f455d7f4c316ab168
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    Prepend batch dimensions if needed.
    You can also give a list of matrices.
    """

    device = 'cuda' if m.is_cuda else 'cpu'  # Note: Using cuda status of m as proxy to decide device

    if type(m) is list:
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

    dim = m.dim()
    n = m.shape[-3]

    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]

    m2 = m.unsqueeze(-2)

    eye = attach_dim(torch.eye(n, device=device).unsqueeze(-2), dim - 3, 1)

    return (m2 * eye).reshape(
        siz0 + torch.Size(torch.tensor(siz1) * n)
    )

def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(torch.Size([1] * n_dim_to_prepend) + v.shape + torch.Size([1] * n_dim_to_append))

def split_spo(triples):
    """ Splits tensor into subject, predicate and object """
    if len(triples.shape) == 2:
        return triples[:, 0], triples[:, 1], triples[:, 2]
    else:
        return triples[:, :, 0], triples[:, :, 1], triples[:, :, 2]

class SparseDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, input):
        input_coal = input.coalesce()
        drop_val = F.dropout(input_coal._values(), self.p, self.training)
        return torch.sparse.FloatTensor(input_coal._indices(), drop_val, input.shape)
class MixedDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dense_dropout = nn.Dropout(p)
        self.sparse_dropout = SparseDropout(p)

    def forward(self, input):
        if input.is_sparse:
            return self.sparse_dropout(input)
        else:
            return self.dense_dropout(input)

class MixedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


def sparse_mx_to_torch_sparse_tensor(sparse_mx): 
    """
    sparse matrix in numpy to tensor 
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def calc_L_hat(adj_matrix):
    
    """
    Symmetric Normalization for Laplace Matrix
    """
    nnodes = adj_matrix.shape[0]
    A = adj_matrix + sp.eye(nnodes) + adj_matrix.transpose()  
    D_vec = A.sum(axis=1).T
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
    x = np.array(D_vec_invsqrt_corr)[0]
    D_invsqrt_corr = sp.diags(x)  # D_(-1/2)
    return sp.eye(nnodes) - D_invsqrt_corr @ A @ D_invsqrt_corr, D_invsqrt_corr @ A @ D_invsqrt_corr

def calc_Lself_hat(adj_matrix):
    nnodes = adj_matrix.shape[0]
    A = adj_matrix  
    D_vec = A.sum(axis=1).T
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
    x = np.array(D_vec_invsqrt_corr)[0]
    D_invsqrt_corr = sp.diags(x)  # D_(-1/2)
    return sp.eye(nnodes) - D_invsqrt_corr @ A @ D_invsqrt_corr, D_invsqrt_corr @ A @ D_invsqrt_corr


def calculate_totalvariation(a_r, l_r, x, num_nodes):
    ft = torch.mm(l_r, x)
    f = torch.mm(ft.t(), x)
    num_edge = a_r._nnz()
    miu = f.trace() / num_edge  # totalvariation
    if math.isnan(miu):
        miu = 0
    return miu

def calculate_relational_totalvariation(num_relations,adj_r,laplace_r,output, num_nodes):
    totalvariation = []
    for r in range(0, num_relations):
        a = adj_r[r].data.to(device)
        l = laplace_r[r].data.to(device)
        miu = calculate_totalvariation(a, l, output, num_nodes)
        totalvariation.append(miu)
        del a, l, miu
    w = torch.reshape(torch.tensor(totalvariation), ((num_relations), 1)).to(device)
    

    return w


def svm_test(X, y, test_sizes=(0.2, 0.4, 0.6, 0.8), repeat=10):
    random_states = [182318 + i for i in range(repeat)]
    result_macro_f1_list = []
    result_micro_f1_list = []
    for test_size in test_sizes:
        macro_f1_list = []
        micro_f1_list = []
        for i in range(repeat):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=True, random_state=random_states[i])
            svm = LinearSVC(dual=False)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            micro_f1 = f1_score(y_test, y_pred, average='micro')
            macro_f1_list.append(macro_f1)
            micro_f1_list.append(micro_f1)
        result_macro_f1_list.append((np.mean(macro_f1_list), np.std(macro_f1_list)))
        result_micro_f1_list.append((np.mean(micro_f1_list), np.std(micro_f1_list)))
    return result_macro_f1_list, result_micro_f1_list


def evaluate_results_nc(embeddings, labels, num_classes):
    print('SVM test')
    svm_macro_f1_list, svm_micro_f1_list = svm_test(embeddings, labels)
    print('Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(macro_f1_mean, macro_f1_std, train_size) for
                                    (macro_f1_mean, macro_f1_std), train_size in
                                    zip(svm_macro_f1_list, [0.8, 0.6, 0.4, 0.2])]))
    print('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(micro_f1_mean, micro_f1_std, train_size) for
                                    (micro_f1_mean, micro_f1_std), train_size in
                                    zip(svm_micro_f1_list, [0.8, 0.6, 0.4, 0.2])]))
    

    return svm_macro_f1_list, svm_micro_f1_list


class OneLayerNet(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super(OneLayerNet, self).__init__()
        self.dropout = dropout
        self.onelayernet = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(input_size, output_size),
        )

    def forward(self, x):
        y_pred = self.onelayernet(x)
        return y_pred

class MLP(nn.Module):
    def __init__(self, input_d, hidden_d, output_d, num_layers, dropout, norm, init_activate):
        super().__init__()

        self.init_activate = init_activate
        self.norm = norm
        self.dropout = dropout
        self.layers = nn.ModuleList([])
        self.num_layers = num_layers

        if num_layers == 1:
            self.layers.append(nn.Linear(input_d, output_d))
        elif num_layers > 1:
            self.layers.append(nn.Linear(input_d, hidden_d))
            for k in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_d, hidden_d))
            self.layers.append(nn.Linear(hidden_d, output_d))

        self.norm_cnt = num_layers - 1 + int(init_activate)  # how many norm layers we have
        if norm == "batch":
            self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_d) for _ in range(self.norm_cnt)])
        elif norm == "layer":
            self.norms = nn.ModuleList([nn.LayerNorm(hidden_d) for _ in range(self.norm_cnt)])

        self.reset_params()

    def reset_params(self):
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight.data)
            # nn.init.constant_(layer.bias.data, 0)


    def activate(self, x):
        if self.norm != "none":
            x = self.norms[self.cur_norm_idx](x)  # use the last norm layer
            self.cur_norm_idx += 1
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return x

    def forward(self, x):
        self.cur_norm_idx = 0

        if self.init_activate:
            x = self.activate(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.num_layers == 1:
                x = F.relu(x)
            if i != len(self.layers) - 1:  # do not activate in the last layer
                x = self.activate(x)

        return x