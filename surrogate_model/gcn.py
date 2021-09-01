# More details can be found in https://github.com/tkipf/pygcn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import pickle
import random
import itertools
import time
import os
import argparse
import networkx
from utils import *
from sklearn.model_selection import train_test_split
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
setup_seed(123)

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

    
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, train_flag=True):
        x = F.relu(self.gc1(x, adj))
        if train_flag:
            train_flag = self.training
        output = F.dropout(x, self.dropout, training=train_flag)
        output = self.gc2(output, adj)
        return output

def main(opts):
    dataset= opts['dataset']
    gpu_id = opts['gpu']
    connect = opts['connect']
    suffix = opts['suffix']
    lr = opts['lr']
    wd = opts['wd']
    nepochs = opts['nepochs']
    dropout = opts['dropout']
    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    adj, features, labels_np = load_npz(f'datasets/{dataset}.npz')
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) + sp.eye(adj.shape[0])
    if connect:
        lcc = largest_connected_components(adj)
        adj = adj[lcc][:, lcc]
        features = features[lcc]
        labels_np = labels_np[lcc]
        n = adj.shape[0]
        print('Nodes num:',n)
#     adj_tensor = normalize(adj)
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    adj_tensor =normalize_tensor(adj_tensor)
    feat = torch.from_numpy(features.todense().astype('double')).float().to(device)
    labels = torch.LongTensor(labels_np).to(device)

    dur = []
    stopper = EarlyStopping(patience=500)
    save_file = 'useful_checkpoint/surrogate_model_gcn/' + dataset + '_' + suffix
    
    mask = np.arange(labels.shape[0])
    train_mask, val_mask, test_mask = train_val_test_split_tabular(mask,train_size=0.64, val_size=0.16, test_size=0.2, random_state=123)
    print('train, val, test:', train_mask.shape, val_mask.shape, test_mask.shape)
    net = GCN(features.shape[1], 64, labels.max().item() + 1, dropout).float().to(device)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    print("lr %.4f, weight_decay %.4f"%(lr, wd))
    
    for epoch in range(nepochs):
        if epoch >=3:
            t0 = time.time()
        _, logits = net(feat, adj_tensor)
        logp = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >=3:
            dur.append(time.time() - t0)
        
        train_acc = accuracy(logp[train_mask], labels[train_mask])
        val_acc = accuracy(logp[val_mask], labels[val_mask])
        print("Epoch {:05d} | Loss {:.4f} | Train Acc {:.5f} | Val Acc: {:.5f} | Time(s) {:.4f}".format(
                        epoch, loss.item(), train_acc, val_acc, np.mean(dur)))
        if stopper.step(val_acc, net, save_file):   
            break
        del loss, logits
    net.load_state_dict(torch.load(save_file+'_checkpoint.pt'))
    val_acc = stopper.best_score
    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    logits = net(feat, adj_tensor,train_flag=False)
    logp = F.log_softmax(logits, dim=1)
    new_val_acc = accuracy(logp[val_mask], labels[val_mask])
    test_acc = accuracy(logp[test_mask], labels[test_mask])
    train_acc = accuracy(logp[train_mask], labels[train_mask])
    acc = accuracy(logp, labels)
    
    print("Train accuracy {:.4%}".format(train_acc))
    print("Validate accuracy ori {:.4%}, new {:.4%}".format(val_acc,new_val_acc))
    print("Test accuracy {:.4%}".format(test_acc))
    print("ALL accuracy {:.4%}".format(acc))

if __name__ == '__main__':
    setup_seed(123)
    parser = argparse.ArgumentParser(description='GCN')

    # configure
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--gpu', type=str, default="1", help='GPU ID')
    parser.add_argument('--connect', default=True, type=bool, help='lcc')
    parser.add_argument('--suffix', type=str, default='_', help='suffix of the checkpoint')

    # dataset
    parser.add_argument('--dataset', default='citeseer', help='dataset to use')
    parser.add_argument('--optimizer', choices=['Adam','SGD', 'RMSprop','Adadelta'], default='RMSprop',
                        help='optimizer')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--wd', default=0., type=float, help='weight decay')
    
    parser.add_argument('--nepochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    args = parser.parse_args()
    opts = args.__dict__.copy()
    print(opts)
    main(opts)


'''
nohup python -u surrogate_gcn.py --dataset 10k_ogbproducts --connect True --suffix lr1e-3 --lr 1e-3 --nepochs 10000 --gpu 0 --dropout 0 > log/surro_gcn_ogbproducts.log  2>&1 &
'''
