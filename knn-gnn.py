import argparse
import torch
import scipy.sparse as sp
import torch.optim as optim
from tqdm import tqdm
from utils import *
#import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv,SAGEConv
import torch.nn as nn
from scipy import spatial
from train_eval import *
from datasets import *
from datasets1 import *
from math import dist
from dataset import CustomDataset
from torch import FloatTensor
import numpy as np
from dgl import ops
from dgl.nn.functional import edge_softmax
from torch.cuda.amp import autocast, GradScaler
import warnings
from uitils import accuracy
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--random_splits', type=bool, default=True)
parser.add_argument('--runs', type=int, default=2);
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--early_stopping', type=int, default=200)
parser.add_argument('--hidden', type=int, default=48)
parser.add_argument('--dropout1', type=float, default=0)
parser.add_argument('--dropout2', type=float, default=0)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--save_path', type=str, default=None)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--output_heads', type=int, default=1)
parser.add_argument('--kernel', type=int, default=3)
parser.add_argument('--embedding', type=int, default=16)
parser.add_argument('--final', type=int, default=16)
parser.add_argument('--proj', type=int, default=4)
parser.add_argument('--model', type=str, default='NLMLP6')
parser.add_argument('--num_warmup_steps', type=int, default=None,
                        help='If None, warmup_proportion is used instead.')
parser.add_argument('--warmup_proportion', type=float, default=0, help='Only used if num_warmup_steps is None.')
parser.add_argument('--hidden_dim', type=int, default=64)
# node feature augmentation
parser.add_argument('--use_sgc_features', default=False, action='store_true')
parser.add_argument('--use_identity_features', default=False, action='store_true')
parser.add_argument('--use_adjacency_features', default=False, action='store_true')
parser.add_argument('--do_not_use_original_features', default=False, action='store_true')
parser.add_argument('--save_dir', type=str, default='experiments', help='Base directory for saving information.')
parser.add_argument('--name', type=str, default=None, help='Experiment name. If None, model name is used.')
parser.add_argument('--num_runs', type=int, default=2)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--amp', default=False, action='store_true')
parser.add_argument('--verbose', default=False, action='store_true')
parser.add_argument('--normalization', type=str, default='LayerNorm', choices=['None', 'LayerNorm', 'BatchNorm'])
parser.add_argument('--dropout', type=float, default=0.8)

parser.add_argument('--num_steps', type=int, default=1000)

args = parser.parse_args()
print(args)

NORMALIZATION = {
    'None': nn.Identity,
    'LayerNorm': nn.LayerNorm,
    'BatchNorm': nn.BatchNorm1d
}
class Net(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,normalization,dropout):
        super(Net, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)

    def forward(self,graph,x):
        x_start, edge_index = x, graph
        x, edge_index = x_start, edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.softmax(x, dim=1)

        return x




class KNNGNN(torch.nn.Module):
    def __init__(self,dataset):
        super(KNNGNN, self).__init__()
        self.lin1 = nn.Linear(dataset.num_features, args.hidden)
        self.gatv1 = GATConv(
            dataset.num_features,
            args.hidden,
            heads=args.heads,
            dropout=args.dropout1,add_self_loops=False)
        self.gatv2 = GATConv(
            args.hidden * args.heads,
            args.hidden,
            heads=args.output_heads,
            concat=False,
            dropout=args.dropout1,add_self_loops=False)
        self.proj = nn.Linear(2 * args.hidden, args.proj)
        self.output_normalization = nn.LayerNorm(2 * args.hidden)
        self.lin = nn.Linear(2 * args.hidden, dataset.num_classes)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.gatv1.reset_parameters()
        self.gatv2.reset_parameters()
        self.proj.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_MLP = F.dropout(x, p=args.dropout1, training=self.training)
        x1 = F.relu(self.lin1(x_MLP))
        x_GAT = F.relu(self.gatv1(x, edge_index))
        x2 = F.dropout(x_GAT, p=args.dropout1, training=self.training)
        x2 = self.gatv2(x2, edge_index)
        final = torch.cat([x1, x2], dim=1)
        g_score = self.proj(final)  # [num_nodes, 1]
        sim_marix = []
        tree = spatial.KDTree(g_score.cpu().detach().numpy())
        # 查询最近的点
        for i in g_score:
            sim_marix.append(torch.mean(final[tree.query(torch.tensor(i), k=35)[1]], dim=0))

        sim_marix = torch.stack(sim_marix).to(device)
        out = self.output_normalization(sim_marix)
        out = torch.cat([x1, sim_marix], dim=1)
        out = self.lin(out)
        return F.log_softmax(out, dim=1),g_score

if args.dataset == "Cora" or args.dataset == "CiteSeer" or args.dataset == "PubMed":
    dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
    permute_masks = random_planetoid_splits if args.random_splits else None
    print("Data:", dataset)
    if args.model=="KNNGNN":
        Net = KNNGNN
    else:
        print("Please choose a correct model!")
    run(dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay, args.early_stopping, permute_masks, lcc=False)
elif args.dataset == "chameleon" or args.dataset == "squirrel" or args.dataset == "film" or args.dataset == "cornell" or args.dataset == "texas" or args.dataset == "wisconsin":
    dataset = get_disassortative_dataset(args.dataset)
    permute_masks = random_disassortative_splits
    print("Data:", dataset)
    if args.model=="KNNGNN":
        Net = KNNGNN
    else:
        print("Please choose a correct model!")
    run_disassortative_dataset(dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay, args.early_stopping, permute_masks, lcc=False)


