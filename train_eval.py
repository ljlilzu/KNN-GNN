from __future__ import division

import time
from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
import numpy as np
from torch_geometric.utils import *
import networkx as nx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, lcc_mask):
    # * 0.6 labels for training
    # * 0.2 labels for validation
    # * 0.2 labels for testing

    indices = []
    if lcc_mask is not None:
        for i in range(num_classes):
            index = (data.y[lcc_mask] == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
    else:
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

    train_index = torch.cat([i[:int(len(i) * 0.6)] for i in indices], dim=0)

    val_index = torch.cat([i[int(len(i) * 0.6):int(len(i) * 0.8)] for i in indices], dim=0)
    test_index = torch.cat([i[int(len(i) * 0.8):] for i in indices], dim=0)

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(test_index, size=data.num_nodes)
    return data


def random_disassortative_splits(data, num_classes):
    # * 0.6 labels for training
    # * 0.2 labels for validation
    # * 0.2 labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:int(len(i) * 0.6)] for i in indices], dim=0)

    val_index = torch.cat([i[int(len(i) * 0.6):int(len(i) * 0.8)] for i in indices], dim=0)
    test_index = torch.cat([i[int(len(i) * 0.8):] for i in indices], dim=0)

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(test_index, size=data.num_nodes)
    return data


def run(dataset, model, runs, epochs, lr, weight_decay, early_stopping,
        permute_masks=None, logger=None, lcc=False, save_path=None):
    durations = []
    val_acc = []
    test_acc_basedonaccs = []

    lcc_mask = None
    if lcc:  # select largest connected component
        data_ori = dataset[0]
        data_nx = to_networkx(data_ori)
        data_nx = data_nx.to_undirected()
        print("Original #nodes:", data_nx.number_of_nodes())
        data_nx = max(nx.connected_component_subgraphs(data_nx), key=len)
        print("#Nodes after lcc:", data_nx.number_of_nodes())
        lcc_mask = list(data_nx.nodes)

    data = dataset[0]

    for _ in range(runs):
        if permute_masks is not None:
            data = permute_masks(data, dataset.num_classes, lcc_mask)
        data = data.to(device)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        test_acc = 0
        test_acc_based_on_acc = 0
        val_loss_history = []
        curr_step = 0
        act_epoch = 0
        best_val_acc = 0


        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()

        val_acc.append(best_val_acc)
        test_acc_basedonaccs.append(test_acc_based_on_acc)
        durations.append(t_end - t_start)

    duration, val_acc, test_acc_basedonacc = tensor(durations), tensor(val_acc), tensor(test_acc_basedonaccs)

    print('Val Acc: {:.4f}, Test Accuracy Based on Val Acc: {:.4f} ± {:.4f}, Duration: {:.4f}'.
          format(val_acc.mean().item(),
                 test_acc_basedonacc.mean().item(),
                 test_acc_basedonacc.std().item(),
                 duration.mean().item()))


def run_disassortative_dataset(dataset, model, runs, epochs, lr, weight_decay, early_stopping,
                               permute_masks=None, logger=None, lcc=False, save_path=None):
    durations = []
    val_acc = []
    test_acc_basedonaccs = []
    data = dataset

    for _ in range(runs):
        if permute_masks is not None:
            data = permute_masks(data, dataset.num_classes)
        data = data.to(device)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        best_val_acc = 0
        test_acc_based_on_acc = 0
        t_end = time.perf_counter()

        val_acc.append(best_val_acc)
        test_acc_basedonaccs.append(test_acc_based_on_acc)
        durations.append(t_end - t_start)

    duration, val_acc, test_acc_basedonacc = tensor(durations), tensor(val_acc), tensor(test_acc_basedonaccs)
    print("epoch:", epoch, 'Val Acc: {:.4f}, Test Accuracy Based on Val Acc: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(val_acc.mean().item(),
                 test_acc_basedonacc.max().item(),
                 test_acc_basedonacc.std().item(),
                 duration.mean().item()))


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out,g = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return out,g


def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        logits, a = model(data)
    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs['{}_loss'.format(key)] = loss
        outs['{}_acc'.format(key)] = acc

        print("%s acc:" % key, outs['{}_acc'.format(key)], end=" ")
        print("%s loss:" % key, outs['{}_loss'.format(key)], end=";")
    print()


def train_step(model, dataset, optimizer, scheduler, scaler, amp=False):
    model.train()

    with autocast(enabled=amp):
        logits = model(dataset.edges, dataset.node_features)
        loss = dataset.loss_fn(input=logits[dataset.train_idx], target=dataset.labels[dataset.train_idx])

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    scheduler.step()


@torch.no_grad()
def evaluate1(model, dataset, amp=False):
    model.eval()

    with autocast(enabled=amp):
        logits = model(dataset.edges, dataset.node_features)

    metrics = dataset.compute_metrics(logits)

    return metrics
