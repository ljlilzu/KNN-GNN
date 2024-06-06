import os
import pickle as pkl
import random
import sys
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
root = os.path.split(__file__)[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def get_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def norm_feat(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    row_sum = np.array(features.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    r_inv = np.power(row_sum.astype(np.float), -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
