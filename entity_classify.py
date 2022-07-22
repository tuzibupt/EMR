import json
import time
import pynvml
import gc
import psutil
import random
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from numpy.lib.function_base import append
from model import EMRGNN
from sklearn.metrics import f1_score, accuracy_score, recall_score
from scipy import sparse
import dgl
from dgl.nn.pytorch import RelGraphConv
import os
from os import link
from utils import evaluate_results_nc
from scripts.data_loader import data_loader


sys.path.append('../../')
pynvml.nvmlInit()
def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)


def evaluate(model_pred, labels):
    pred_result = model_pred.argmax(dim=1)
    labels = labels.cpu()
    pred_result = pred_result.cpu()
    micro = f1_score(labels, pred_result, average='micro')
    macro = f1_score(labels, pred_result, average='macro')
    acc = accuracy_score(labels, pred_result)
    macro_recall = recall_score(labels, pred_result, average='macro')
    micro_recall = recall_score(labels, pred_result, average='micro')
    return micro, macro, acc, macro_recall, micro_recall


def multi_evaluate(model_pred, labels):
    model_pred = torch.sigmoid(model_pred)
    accuracy_th = 0.5
    pred_result = model_pred > accuracy_th
    pred_result = pred_result.float()
    labels = labels.cpu()
    pred_result = pred_result.cpu()
    micro = f1_score(labels, pred_result, average='micro')
    macro = f1_score(labels, pred_result, average='macro')
    acc = accuracy_score(labels, pred_result)
    macro_recall = recall_score(labels, pred_result, average='macro')
    micro_recall = recall_score(labels, pred_result, average='micro')
    return micro, macro, acc, macro_recall, micro_recall


class EntityClassify(EMRGNN):
    def build_input_layer(self):
        # return None
        return nn.ModuleList([nn.Linear(in_dim, self.h_dim, bias=True) for in_dim in self.in_dims])

def main(args):
    seed_it(args.seed)
    args_dataset = args.dataset.upper()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    if args.gpu >= 0:
        handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_begin = meminfo.used
    #print("begin:", gpu_begin)
    dataset = ['DBLP', 'ACM', 'MUTAG', 'BGS']# more dataset
    if args_dataset in dataset:
        dataset = None
    else:
        raise ValueError()

    LOSS = F.cross_entropy
    folder = 'data/'+args_dataset
    dl = data_loader(folder)
    all_data = {}
    
    for etype in dl.links['meta']:
        etype_info = dl.links['meta'][etype]
        metrix = dl.links['data'][etype]
        if args_dataset in ['MUTAG','BGS']:  # for multi-relation datasets
            all_data[(etype_info[0], 'link'+str(etype), etype_info[1])] = (
                sparse.find(metrix)[0]-dl.nodes['shift'][etype_info[0]], sparse.find(metrix)[1]-dl.nodes['shift'][etype_info[1]])
        else: # for heterogeneous datasets
            all_data[(etype_info[0], 'link', etype_info[1])] = (
                sparse.find(metrix)[0]-dl.nodes['shift'][etype_info[0]], sparse.find(metrix)[1]-dl.nodes['shift'][etype_info[1]])


    hg = dgl.heterograph(all_data)
    category_id = list(dl.labels_train['count'].keys())[0]
    train_idx = np.nonzero(dl.labels_train['mask'])[0]
    test_idx = np.nonzero(dl.labels_test['mask'])[0]
    labels = torch.LongTensor(
            dl.labels_train['data']+dl.labels_test['data']).argmax(dim=1)
    num_classes = dl.labels_test['num_classes']
    num_rels = len(hg.canonical_etypes)
    EVALUATE = evaluate

    # split dataset into train, validate, test; the number of training samples can be changed by changing the hyperparameter radio
    if args.validation:
        if args_dataset in ['MUTAG','BGS']:
            train_idx = train_idx[len(train_idx) // 5:]
            val_idx = train_idx[:len(train_idx) // 5]
        else:
            train_idx = np.loadtxt(folder+"/train_index.txt", dtype='int')
            train_idx = train_idx[:int(len(train_idx)*args.radio)] 
            val_idx = np.loadtxt(folder+"/val_index.txt", dtype='int')     


    # calculate norm for each edge type and store in edge
    for canonical_etype in hg.canonical_etypes:
        u, v, eid = hg.all_edges(form='all', etype=canonical_etype)
        _, inverse_index, count = torch.unique(
            v, return_inverse=True, return_counts=True)
        degrees = count[inverse_index]
        norm = torch.ones(eid.shape[0]).float() / degrees.float()
        norm = norm.unsqueeze(1)
        hg.edges[canonical_etype].data['norm'] = norm

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda:'+str(args.gpu) if use_cuda else 'cpu')
    torch.cuda.set_device(args.gpu)
    g = dgl.to_homogeneous(hg, edata=['norm'])
    num_nodes = g.number_of_nodes()
    node_ids = torch.arange(num_nodes)
    edge_norm = g.edata['norm']
    edge_type = g.edata[dgl.ETYPE].long()
    num_relations = len(dl.links['count'])

    # find out the target node ids in g
    node_tids = g.ndata[dgl.NTYPE]
    loc = (node_tids == category_id)
    target_idx = node_ids[loc]
     
    #Construct triples [source,relation,target]
    temp = 0
    tri_list = []
    for k, value in all_data.items():
        i = value[0]
        i_1 = torch.tensor(i).reshape(len(i), 1)
        j = value[1]
        j_1 = torch.tensor(j).reshape(len(j), 1)
        # id-g_id
        c_id1 = k[0]
        loc = (node_tids == c_id1)
        t_idx_1 = node_ids[loc]
        step1 = t_idx_1[0]
        i_1 = i_1 + step1
        c_id2 = k[2]
        loc = (node_tids == c_id2)
        t_idx_2 = node_ids[loc]
        step2 = t_idx_2[0]
        j_1 = j_1 + step2
        id = torch.empty(size=(len(i), 1), dtype=torch.long).fill_(temp)
        tri = torch.cat([i_1, id, j_1], dim=1)
        tri_list.append(tri)
        temp = temp + 1
    triples = tri_list[0]
    for t in range(1, len(all_data)):
        triples = torch.cat([triples, tri_list[t]], dim=0)
    triples = triples.to(device)
    edge_type = edge_type.to(device)
    edge_norm = edge_norm.to(device)
    labels = labels.to(device)

    
    # Initialize node features in different ways
    features_list = []
    in_dims = [args.n_hidden]
    if args_dataset not in ['BGS']:
        for i in range(len(dl.nodes['count'])):
            th = dl.nodes['attr'][i]
            if th is None:
                features_list.append(np.eye(dl.nodes['count'][i]))
            else:
                features_list.append(th)
        features_list = [mat2tensor(features).to(device)
                         for features in features_list]
        feats_type = args.feats_type
        in_dims = []
        if feats_type == 0:
            in_dims = [features.shape[1] for features in features_list]
        elif feats_type == 1 or feats_type == 5:
            save = 0 if feats_type == 1 else 2
            in_dims = []
            for i in range(0, len(features_list)):
                if i == save:
                    in_dims.append(features_list[i].shape[1])
                else:
                    in_dims.append(10)
                    features_list[i] = torch.zeros(
                        (features_list[i].shape[0], 10)).to(device)
        elif feats_type == 2 or feats_type == 4:
            save = feats_type - 2
            in_dims = [features.shape[0] for features in features_list]
            for i in range(0, len(features_list)):
                if i == save:
                    in_dims[i] = features_list[i].shape[1]
                    continue
                dim = features_list[i].shape[0]
                indices = np.vstack((np.arange(dim), np.arange(dim)))
                indices = torch.LongTensor(indices)
                values = torch.FloatTensor(np.ones(dim))
                features_list[i] = torch.sparse.FloatTensor(
                    indices, values, torch.Size([dim, dim])).to(device)
        elif feats_type == 3:
            in_dims = [features.shape[0] for features in features_list]
            for i in range(len(features_list)):
                dim = features_list[i].shape[0]
                indices = np.vstack((np.arange(dim), np.arange(dim)))
                indices = torch.LongTensor(indices)
                values = torch.FloatTensor(np.ones(dim))
                features_list[i] = torch.sparse.FloatTensor(
                    indices, values, torch.Size([dim, dim])).to(device)


    model = EntityClassify(in_dims,
                           args.n_hidden,
                           num_classes,
                           triples = triples,
                           num_classes = num_classes,
                           dropout = args.dropout,
                           lamda_1 = args.lamda1,
                           lamda_2 = args.lamda2,
                           transf_type = args.transf_type,
                           threshold = args.threshold,
                           num_nodes=num_nodes,
                            n_mlp_layer = args.n_mlp_layer,
                            threshold_c = args.threshold_c,
                            hm_dim = args.hm_dim,
                            args_dataset = args_dataset)

    model.to(device)
    g = g.to('cuda:%d' % args.gpu)

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []
    save_dict_micro = {}
    save_dict_macro = {}
    save_dict_acc = {}
    best_result_micro = 0
    best_result_macro = 0
    best_epoch_micro = 0
    best_epoch_macro = 0
    best_result_acc =0
    best_epoch_acc = 0
    
    model.train()
    for epoch in range(args.n_epochs):
        optimizer.zero_grad()
        t0 = time.time()
        #logits,embeddings = model(g, features_list, edge_type, edge_norm,  num_nodes, num_relations)
        logits,embeddings,rel_cor = model(g, features_list, edge_type, edge_norm,  num_nodes, num_relations,args_dataset)
        logits = logits[target_idx]
        loss = LOSS(logits[train_idx], labels[train_idx])
        t1 = time.time()
        loss.backward()
        optimizer.step()
        t2 = time.time()
        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)

        val_loss = LOSS(logits[val_idx], labels[val_idx])
        train_micro, train_macro, train_acc, trian_macro_recall, trian_micro_recall = EVALUATE(
            logits[train_idx], labels[train_idx])
        valid_micro, valid_macro, valid_acc, val_macro_recall, val_micro_recall = EVALUATE(
            logits[val_idx], labels[val_idx])
        test_loss = LOSS(logits[test_idx], labels[test_idx])
        test_micro, test_macro, test_acc, test_macro_recall, test_micro_recall = EVALUATE(
                logits[test_idx], labels[test_idx])
               
        if valid_micro > best_result_micro:
            save_dict_micro = model.state_dict()
            best_result_micro = valid_micro
            best_epoch_micro = epoch
        if valid_macro > best_result_macro:
            save_dict_macro = model.state_dict()
            best_result_macro = valid_macro
            best_epoch_macro = epoch
        if valid_acc > best_result_acc:
            save_dict_acc = model.state_dict()
            best_result_acc = valid_acc
            best_epoch_acc = epoch
        print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Train Loss: {:.4f}| Backward Time(s) {:.4f}|Validation loss: {:.4f}".
              format(epoch, forward_time[-1],loss.item(),  backward_time[-1], val_loss.item()))
     
    model.eval()
    result = [save_dict_micro, save_dict_macro,save_dict_acc]
    torch.cuda.empty_cache()
    with torch.no_grad():
        for i in range(1):
            if i == 0:
                print("Best Micro At:"+str(best_epoch_micro))
            elif i == 1:
                print("Best Macro At:"+str(best_epoch_macro))
            else:
                print("Best acc At:"+str(best_epoch_acc))
            model.load_state_dict(result[i])
            t0 = time.time()
            #logits, embeddings = model.forward(g, features_list, edge_type, edge_norm, num_nodes, num_relations)
            logits, embeddings, rel_cor = model.forward(g, features_list, edge_type, edge_norm, num_nodes, num_relations,args_dataset)
            t1 = time.time()
            print("test time:"+str(t1-t0))
            logits = logits[target_idx]
            test_loss = LOSS(logits[test_idx], labels[test_idx]) 
            test_logits = logits[test_idx]
            test_micro, test_macro, test_acc, test_macro_recall, test_micro_recall = EVALUATE(
                logits[test_idx], labels[test_idx])
            print( test_acc, test_macro_recall)
            print("Test acc: {:.4f} | Test_recall: {:.4f}".format(test_acc, test_macro_recall))

    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_end = meminfo.used
    print("test end:", gpu_end)
    print("net gpu usage:", (gpu_end-gpu_begin)/1024/1024, 'MB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument('--feats-type', type=int, default=3,
                        help='Type of the node features used. ' +
                        '0 - loaded features; ' +
                        '1 - only target node features (zero vec for others); ' +
                        '2 - only target node features (id vec for others); ' +
                        '3 - all id vec. Default is 2;' +
                        '4 - only term features (id vec for others);' +
                        '5 - only term features (zero vec for others).')
    parser.add_argument("--dropout", type=float, default=0,
                        help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=64,
                        help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-layers", type=int, default=3,
                        help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=150,
                        help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use")
    parser.add_argument("--l2norm", type=float, default=1,
                        help="l2 norm coef")
    parser.add_argument("--use-self-loop", default=False, action='store_true',
                        help="include self feature as a special relation")
    parser.add_argument('--transf_type', type=int, default=2,
                    help='Type of the node features initialization and transformation' +
                    '0 - kaiming_normal features + linear transformation; ' +
                    '1 - no  transformation' +
                    '2 - mlp transformation).' 
                  )
    parser.add_argument("--lamda1", type=float, default=20,
                        help="lamda1, default: 20 [use all]")
    parser.add_argument("--lamda2", type=float, default=30,
                        help="lamda2, default: 30 [use all]")
    parser.add_argument("--threshold", type=float, default=1/28,
                        help="threshold for model para, default: 1/28 [use all]")
    parser.add_argument("--hm_dim", type=int, default=32,
                        help="number of hidden units")
    parser.add_argument("--n_mlp_layer", type=int, default=1,
                        help="mlp layer for model para, default: 1 [use all]")
    parser.add_argument("--threshold_c", type=float, default=0.1,
                        help="threshold for model para, default: 0.1 [use all]")
    parser.add_argument("--radio", type=float, default=1.0,
                        help="radio for train sets, default: 1.0 [use all]")
    parser.add_argument("--seed", type=int, default=1234,
                        help="random seed: 1234 [use all]") 
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)
    args = parser.parse_args()
    main(args)
