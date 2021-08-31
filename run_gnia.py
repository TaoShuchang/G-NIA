import torch
import time
import sys
import os
import math
import argparse
import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data as Data
np_load_old = np.load
np.aload = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
from gnia import GNIA
sys.path.append('..')
from utils import *
from surrogate_model.gcn import GCN
from surrogate_model.gat import GAT, LayerType
from surrogate_model.appnp import PPNP, PPRPowerIteration

setup_seed(123)

def main(opts):
    # hyperparameters
    gpu_id = opts['gpu']
    seed = opts['seed']
    surro_type = opts['surro_type']
    victim_type = opts['victim_type']
    dataset= opts['dataset']
    connect = opts['connect']
    multi = opts['multiedge']
    discrete = opts['discrete']
    suffix = opts['suffix']
    attr_tau = float(opts['attrtau']) if opts['attrtau']!=None else opts['attrtau']
    edge_tau = float(opts['edgetau']) if opts['edgetau']!=None else opts['edgetau']
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
    # local_rank = opts['local_rank']
    # torch.cuda.set_device(local_rank)
    # dist.init_process_group(backend='nccl') 

    surro_save_file = 'checkpoint/surrogate_model/' + dataset + '_' + surro_type
    victim_save_file = 'checkpoint/surrogate_model/' + dataset + '_' + surro_type
    ckpt_save_dirs = 'checkpoint/' + surro_type + '_gnia/'
    output_save_dirs = 'output/' + surro_type + '_gnia/'
    model_save_file = ckpt_save_dirs + dataset + '_' + suffix
    if not os.path.exists(ckpt_save_dirs):
        os.makedirs(ckpt_save_dirs)
    if not os.path.exists(output_save_dirs):
        os.makedirs(output_save_dirs)

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocessing data
    adj, features, labels_np = load_npz(f'datasets/{dataset}.npz')
    n = adj.shape[0]
    nc = labels_np.max()+1
    nfeat = features.shape[1]
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) + sp.eye(n)
    adj[adj > 1] = 1
    if connect:
        lcc = largest_connected_components(adj)
        adj = adj[lcc][:, lcc]
        features = features[lcc]
        labels_np = labels_np[lcc]
        n = adj.shape[0]
        print('Nodes num:',n)

    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    nor_adj_tensor = normalize_tensor(adj_tensor)

    feat = torch.from_numpy(features.todense().astype('double')).float().to(device)
    feat_max = feat.max(0).values
    feat_min = feat.min(0).values
    labels = torch.LongTensor(labels_np).to(device)
    degree = adj.sum(1)
    deg = torch.FloatTensor(degree).flatten().to(device)
    feat_num = int(features.sum(1).mean())
    eps_threshold = [epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * steps / epsilon_decay) for steps in range(total_steps)]
    
    split = np.aload('datasets/' + dataset+ '_split.npy').item()
    train_mask = split['train']
    val_mask = split['val']
    test_mask = split['test']

    print("Surrogate GNN Model:", surro_type)
    print("Evaluation GNN Model:", victim_type)

    # Surrogate model
    if surro_type == 'gcn':
        surro_net = GCN(features.shape[1], 64, labels.max().item() + 1, 0.5).float().to(device)

    elif surro_type == 'gat':
        surro_net = GAT(num_of_layers=2, num_heads_per_layer=[nhead, 1], num_features_per_layer=[nfeat, nhid, nc],
            add_skip_connection=False, bias=True, dropout=0.6, 
            layer_type=LayerType.IMP2, log_attention_weights=False).to(device)
    else:
        prop_appnp = PPRPowerIteration(alpha=0.1, niter=10)
        surro_net = PPNP(nfeat, nc, [64], 0.5, prop_appnp).to(device)
    surro_net.load_state_dict(torch.load(surro_save_file+'_checkpoint.pt'))

    # Evalution model
    if victim_type == 'gcn':
        victim_net = GCN(features.shape[1], 64, labels.max().item() + 1, 0.5).float().to(device)
    elif victim_type == 'gat':
        victim_net = GAT(num_of_layers=2, num_heads_per_layer=[nhead, 1], num_features_per_layer=[nfeat, nhid, nc],
            add_skip_connection=False, bias=True, dropout=0.6, 
            layer_type=LayerType.IMP2, log_attention_weights=False).to(device)
    else:
        prop_appnp = PPRPowerIteration(alpha=0.1, niter=10)
        victim_net = PPNP(nfeat, nc, [64], 0.5, prop_appnp).to(device)
    victim_net.load_state_dict(torch.load(victim_save_file+'_checkpoint.pt'))

    surro_net.eval()
    victim_net.eval()
    for p in victim_net.parameters():
        p.requires_grad = False
    for p in surro_net.parameters():
        p.requires_grad = False

    if surro_type == 'gcn':
        node_emb = surro_net(feat, nor_adj_tensor)
        W1 = surro_net.gc1.weight.data.detach()
        W2 = surro_net.gc2.weight.data.detach()
    elif surro_type == 'gat':
        adj_topo_tensor = torch.tensor(adj.toarray(), dtype=torch.float, device=device)
        graph_data = (feat, adj_topo_tensor)
        node_emb = surro_net(graph_data)[0]
        W1 = surro_net.gat_net[0].linear_proj.weight.data.detach().t()
        W2 = surro_net.gat_net[1].linear_proj.weight.data.detach().t()
    else:
        node_emb = surro_net(feat, nor_adj_tensor)
        W1 = surro_net.fcs[0].weight.data.detach()
        W2 = surro_net.fcs[1].weight.data.detach().t()
    W = torch.mm(W1, W2).t()    
    
    if victim_type == 'gat':
        graph_data = (feat, adj_topo_tensor)
        logits = victim_net(graph_data)[0]
    else:
        logits = victim_net(feat, nor_adj_tensor)
    sec = worst_case_class(logits, labels_np)
    logp = F.log_softmax(logits, dim=1)
    acc = accuracy(logp, labels)
    print('Acc:',acc)
    print('Train Acc:',accuracy(logp[train_mask], labels[train_mask]))
    print('Valid Acc:',accuracy(logp[val_mask], labels[val_mask]))
    print('Test Acc:',accuracy(logp[test_mask], labels[test_mask]))
    
    # Initialization
    model = GNIA(labels, nfeat, W1, W2, discrete, device, feat_min=feat_min, feat_max=feat_max, feat_num=feat_num, attr_tau=attr_tau, edge_tau=edge_tau).to(device)
    # model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    stopper = EarlyStop_loss(patience=patience)

    if opts['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=lr)
    elif opts['optimizer'] == 'RMSprop':
        optimizer = torch.optim.RMSprop([{'params': model.parameters()}], lr=lr, weight_decay=0)
    else: 
        raise ValueError('Unsupported argument for the optimizer')

    
    x = torch.LongTensor(train_mask)
    y = labels[train_mask].to(torch.device('cpu'))
    torch_dataset = Data.TensorDataset(x,y)
    # train_sampler = Data.distributed.DistributedSampler(
    #     torch_dataset,
    #     num_replicas=2,
    #     rank=local_rank,
    # )
    # batch_loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, num_workers=24, sampler=train_sampler)
    batch_loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, num_workers=24)
    
    if st_epoch != 0:
        model.load_state_dict(torch.load(model_save_file+'_checkpoint.pt'))
        stopper.best_score = best_score
        stopper.counter = counter
    # Training and Validation
    for epoch in range(st_epoch, nepochs):
        training = True
        print("Epoch {:05d}".format(epoch))
        train_atk_success = []
        val_atk_success = []
        train_loss_arr = []
        eps = eps_threshold[epoch] if epoch < len(eps_threshold) else eps_threshold[-1]

        for batch_x,_ in batch_loader:
            loss_arr = []

            for train_batch in batch_x:
                target = np.array([train_batch])
                target_deg = int(sum([degree[i].item() for i in target]))
                budget = int(min(round(target_deg/2), round(degree.mean()))) if multi else 1
                best_wrong_label = sec[target[0]]
                ori = labels_np[target].item()
                one_order_nei = adj[target].nonzero()[1]

                if surro_type== 'gat':
                    one_order_nei, four_order_nei, sub_tar, sub_idx = k_order_nei(adj.toarray(), 3, target)
                    tar_norm_adj = nor_adj_tensor[sub_tar.item()].to_dense()
                    norm_a_target = tar_norm_adj[sub_idx].unsqueeze(1)
                    sub_feat = feat[four_order_nei]
                    sub_adj = adj.toarray()[four_order_nei][:,four_order_nei]
                    sub_adj_tensor = torch.tensor(sub_adj, dtype=torch.float, device=device)
                    inj_feat, disc_score, masked_score_idx  = model(sub_tar, sub_idx, budget, sub_feat, norm_a_target, node_emb[four_order_nei],
                                                W[ori], W[best_wrong_label], train_flag=training,eps=eps)
                    new_feat = torch.cat((sub_feat, inj_feat.unsqueeze(0)), 0)
                else:
                    tar_norm_adj = nor_adj_tensor[target.item()].to_dense()
                    norm_a_target = tar_norm_adj[one_order_nei].unsqueeze(1)
                    inj_feat, disc_score, masked_score_idx = model(target, one_order_nei, budget, feat, norm_a_target, node_emb,
                                                    W[ori], W[best_wrong_label], train_flag=training, eps=eps)
                    new_feat = torch.cat((feat, inj_feat.unsqueeze(0)), 0)

                if victim_type == 'gat':
                    new_adj_tensor = gen_new_adj_topo_tensor(sub_adj_tensor, disc_score, sub_idx, device)
                    new_logits = victim_net((new_feat, new_adj_tensor))[0]
                    loss = F.relu(new_logits[sub_tar,ori] - new_logits[sub_tar,best_wrong_label])
                    new_logp = F.log_softmax(new_logits, dim=1)    
                    train_atk_success.append(int(ori != new_logits[sub_tar].argmax(1).item()))
                else:
                    new_adj_tensor = gen_new_adj_tensor(adj_tensor, disc_score, masked_score_idx, device)
                    new_logits = victim_net(new_feat, normalize_tensor(new_adj_tensor))
                    loss = F.relu(new_logits[target,ori] - new_logits[target,best_wrong_label])
                    new_logp = F.log_softmax(new_logits, dim=1)                    
                    train_atk_success.append(int(ori != new_logits[target].argmax(1).item()))
                loss_arr.append(loss)
            train_loss = np.array(loss_arr).sum()
            optimizer.zero_grad()
            train_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.1)
            optimizer.step()
            train_loss_arr.append((train_loss/len(loss_arr)).detach().cpu().item())
            
            del loss_arr, train_loss
        print('Training set Loss:', np.array(train_loss_arr).mean())
        print('Training set: Attack success rate:', np.array(train_atk_success).mean())
        del train_loss_arr, train_atk_success
        torch.cuda.empty_cache()


        val_loss_arr = []
        training = False
        for val_batch in val_mask:
            target = np.array([val_batch])
            target_deg = int(sum([degree[i].item() for i in target]))
            budget = int(min(round(target_deg/2), round(degree.mean()))) if multi else 1
            best_wrong_label = sec[target[0]]
            ori = labels_np[target].item()
            one_order_nei = adj[target].nonzero()[1]
            if surro_type== 'gat':
                one_order_nei, four_order_nei, sub_tar, sub_idx = k_order_nei(adj.toarray(), 3, target)
                tar_norm_adj = nor_adj_tensor[sub_tar.item()].to_dense()
                norm_a_target = tar_norm_adj[sub_idx].unsqueeze(1)
                # sub_nor_adj_tensor = nor_adj_tensor.to_dense()[four_order_nei][:, four_order_nei]
                sub_feat = feat[four_order_nei]
                sub_adj = adj.toarray()[four_order_nei][:,four_order_nei]
                sub_adj_tensor = torch.tensor(sub_adj, dtype=torch.float, device=device)
                inj_feat, disc_score, masked_score_idx  = model(sub_tar, sub_idx, budget, sub_feat, norm_a_target, node_emb[four_order_nei],
                                            W[ori], W[best_wrong_label], train_flag=training,eps=eps)
                new_feat = torch.cat((sub_feat, inj_feat.unsqueeze(0)), 0)
            else:
                tar_norm_adj = nor_adj_tensor[target.item()].to_dense()
                norm_a_target = tar_norm_adj[one_order_nei].unsqueeze(1)
                inj_feat, disc_score, masked_score_idx = model(target, one_order_nei, budget, feat, norm_a_target, node_emb,
                                                W[ori], W[best_wrong_label], train_flag=training, eps=eps)
                new_feat = torch.cat((feat, inj_feat.unsqueeze(0)), 0)

            if victim_type == 'gat':
                new_adj_tensor = gen_new_adj_topo_tensor(sub_adj_tensor, disc_score, sub_idx, device)
                new_logits = victim_net((new_feat, new_adj_tensor))[0]
                loss = F.relu(new_logits[sub_tar,ori] - new_logits[sub_tar,best_wrong_label])
                new_logp = F.log_softmax(new_logits, dim=1)    
                val_atk_success.append(int(ori != new_logits[sub_tar].argmax(1).item()))
            else:
                new_adj_tensor = gen_new_adj_tensor(adj_tensor, disc_score, masked_score_idx, device)
                new_logits = victim_net(new_feat, normalize_tensor(new_adj_tensor))
                loss = F.relu(new_logits[target,ori] - new_logits[target,best_wrong_label])
                new_logp = F.log_softmax(new_logits, dim=1)                    
                val_atk_success.append(int(ori != new_logits[target].argmax(1).item()))
            val_loss_arr.append(loss.detach().cpu().item())
        print('Validation set Loss:', np.array(val_loss_arr).mean())
        print('Validation set: Attack success rate:', np.array(val_atk_success).mean())

        val_loss = np.array(val_loss_arr).mean()

        if stopper.step(val_loss, model, model_save_file):   
            break
        del val_loss_arr, val_atk_success
        torch.cuda.empty_cache()


    # Test Part
    names = locals()
    training = False
    model.load_state_dict(torch.load(model_save_file+'_checkpoint.pt'))
    for p in model.parameters():
        p.requires_grad = False
    atk_suc = []
    feat_sum = []
    for dset in ['train', 'val', 'test']:
        names[dset + '_atk_suc']  = []
        for batch in names[dset + '_mask']:
            target = np.array([batch])
            target_deg = int(sum([degree[i].item() for i in target]))
            budget = int(min(round(target_deg/2), round(degree.mean()))) if multi else 1
            best_wrong_label = sec[target[0]]
            ori = labels_np[target].item()
            one_order_nei = adj[target].nonzero()[1]

            if surro_type== 'gat':
                one_order_nei, four_order_nei, sub_tar, sub_idx = k_order_nei(adj.toarray(), 3, target)
                tar_norm_adj = nor_adj_tensor[sub_tar.item()].to_dense()
                norm_a_target = tar_norm_adj[sub_idx].unsqueeze(1)
                # sub_nor_adj_tensor = nor_adj_tensor.to_dense()[four_order_nei][:, four_order_nei]
                sub_feat = feat[four_order_nei]
                sub_adj = adj.toarray()[four_order_nei][:,four_order_nei]
                sub_adj_tensor = torch.tensor(sub_adj, dtype=torch.float, device=device)
                inj_feat, disc_score, masked_score_idx  = model(sub_tar, sub_idx, budget, sub_feat, norm_a_target, node_emb[four_order_nei],
                                            W[ori], W[best_wrong_label], train_flag=training,eps=eps)
                new_feat = torch.cat((sub_feat, inj_feat.unsqueeze(0)), 0)
            else:
                tar_norm_adj = nor_adj_tensor[target.item()].to_dense()
                norm_a_target = tar_norm_adj[one_order_nei].unsqueeze(1)
                inj_feat, disc_score, masked_score_idx = model(target, one_order_nei, budget, feat, norm_a_target, node_emb,
                                                W[ori], W[best_wrong_label], train_flag=training, eps=eps)
                new_feat = torch.cat((feat, inj_feat.unsqueeze(0)), 0)

            if victim_type == 'gat':
                new_adj_tensor = gen_new_adj_topo_tensor(sub_adj_tensor, disc_score, sub_idx, device)
                new_logits = victim_net((new_feat, new_adj_tensor))[0]
                loss = F.relu(new_logits[sub_tar,ori] - new_logits[sub_tar,best_wrong_label])
                out_tar = sub_tar
            else:
                new_adj_tensor = gen_new_adj_tensor(adj_tensor, disc_score, masked_score_idx, device)
                new_logits = victim_net(new_feat, normalize_tensor(new_adj_tensor))
                out_tar = target

            print(dset +' Node: %d, Degree: %d' % (batch, target_deg))
            new_logp = F.log_softmax(new_logits, dim=1)
            feat_nz = new_feat[-1].detach().cpu().nonzero()
            edge_nz = disc_score.detach().cpu().nonzero()
            if discrete:
                print('\t Attribute:', feat_nz.shape[0], feat_nz.squeeze())
            print('\t Edge:', edge_nz.shape[0], edge_nz.squeeze())
            print('\t pred: %d, sec: %d, label: %d'%(new_logits[out_tar].argmax(), best_wrong_label, ori))
            
            if ori != new_logp[out_tar].argmax(1).item():
                print("\t Attack successfully!!!")
                names[dset + '_atk_suc'].append(1)
                atk_suc.append(1)
            else:
                print("\t Attack Failed###")
                names[dset + '_atk_suc'].append(0)
                atk_suc.append(0)
            del  new_logits, new_logp
        print('Attack success rate of '+ dset +' set:', np.array(names[dset + '_atk_suc']).mean())
        print('*'*30)

    # np.save('useful_output/' + surro_type + '_gene/' + dataset + '_' + suffix + '_featsum.npy', np.array(feat_sum))
    np.save(output_save_dirs + dataset + '_' + suffix + '_atk_success.npy', np.array(atk_suc))
        



if __name__ == '__main__':
    setup_seed(123)
    parser = argparse.ArgumentParser(description='GNIA')

    # configure
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--gpu', type=str, default="1", help='GPU ID')
    parser.add_argument('--suffix', type=str, default='_', help='suffix of the checkpoint')

    # dataset
    parser.add_argument('--dataset', default='citeseer',help='dataset to use')
    
    parser.add_argument('--surro_type', default='gcn',help='surrogate gnn model')
    parser.add_argument('--victim_type', default='gcn',help='evaluation gnn model')

    # optimization
    parser.add_argument('--optimizer', choices=['Adam','SGD', 'RMSprop'], default='RMSprop',
                        help='optimizer')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--wd', default=0., type=float , help='weight decay')
    
    parser.add_argument('--attrtau', default=None, help='tau of gumbel softmax on attr')
    parser.add_argument('--edgetau', default=None, help='tau of gumbel softmax on edge')
    parser.add_argument('--epsdec', default=50, type=float, help='epsilon decay: coefficient of the gumbel sampling')
    parser.add_argument('--epsst', default=50, type=int, help='epsilon start: coefficient of the gumbel sampling')
    # parser.add_argument('--alpha', default=0.1, type=float, help='coeffiencient of sparsity constraint')
    parser.add_argument('--patience', default=100, type=int, help='patience of early stopping')
    parser.add_argument('--connect', default=False, type=bool, help='lcc')
    parser.add_argument('--multiedge', default=False, type=bool, help='budget of edges connected to injected node')
    
    parser.add_argument('--counter', type=int, default=0, help='counter')
    parser.add_argument('--best_score', type=float, default=0., help='best score')
    parser.add_argument('--st_epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--nepochs', type=int, default=10000, help='number of epochs')
    parser.add_argument('--batchsize', type=int, default=8, help='batchsize')
    parser.add_argument('--k', type=int, default=10, help='the target node to attack')
    parser.add_argument('--local_rank', type=int, default=2, help='DDP local rank')
    
    args = parser.parse_args()
    opts = args.__dict__.copy()
    GAT_para = {'12k_reddit':(4,4), '10k_ogbproducts':(6,6), 'citeseer':(8,8)}
    opts['nhid'], opts['nhead'] = GAT_para[opts['dataset']]
    opts['discrete'] = False if 'k_' in opts['dataset'] else True
    print(opts)
    att_sucess = main(opts) 

'''
nohup python -u run_gnia.py --suffix multi_gcn --multiedge True --nepochs 10000 --lr 1e-5 --connect True --epsst 50 --epsdec 1 --patience 500 --dataset citeseer --attrtau 1 --edgetau 0.01 --surro_type gcn --victim_type gcn --batchsize 32 > log/white_gcn_gnia/citeseer_multi.log 2>&1 &
'''


