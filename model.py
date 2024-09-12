import math
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops, add_self_loops


def weighted_gcn_norm(edge_index, edge_weight=None, num_nodes=None, eps=1., dtype=None):
    fill_value = eps
    num_nodes = int(edge_index.max()) + 1 if num_nodes is None else num_nodes
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)

    edge_index, tmp_edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)
    assert tmp_edge_weight is not None
    edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, norm


class ASPSGC(MessagePassing):
    def __init__(self, in_channels: int, hidden_channels: int, eps: float):
        super(ASPSGC, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.eps = eps
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, K=1):
        edge_index, norm = weighted_gcn_norm(edge_index, edge_weight, x.size(0), eps=self.eps, dtype=x.dtype)

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.lin1(x)
        raw = x
        # propagate
        for k in range(K):
            x = self.propagate(edge_index, x=x, norm=norm)
            x = (1. - self.eps) * raw + x
            raw = x

        return x

    def message(self, x_j, norm):

        return norm.view(-1, 1) * x_j


class Model(nn.Module):
    def __init__(self, num_features, args):
        super(Model, self).__init__()
        self.pri_hop = args.pri_hop
        self.sup_hop = args.sup_hop
        self.global_hop = args.global_hop
        self.tau = args.tau
        self.gcn1 = ASPSGC(num_features, args.num_hidden, args.eps1)
        self.gcn2 = ASPSGC(num_features, args.num_hidden, args.eps2)
        self.reset_parameters()

    def reset_parameters(self):
        self.gcn1.reset_parameters()
        self.gcn2.reset_parameters()

    def forward(self, x, pri_edges, sup_edges, training=True):
        z1 = self.gcn1(x, pri_edges, K=self.pri_hop)
        z2 = self.gcn2(x, sup_edges, K=self.sup_hop) + self.gcn2(x, pri_edges, K=self.global_hop)
        if training:
            return z1, z2
        else:
            embs = (z1 + z2).detach()
            return embs

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        between_sim = f(self.sim(z1, z2))
        refl_sim1 = f(self.sim(z1, z1))
        refl_sim2 = f(self.sim(z2, z2))
        return (-torch.log(
            between_sim.diag()
            / (refl_sim1.sum(1) + between_sim.sum(1) - refl_sim1.diag() + refl_sim2.sum(1) - refl_sim2.diag()))).mean()

    def loss(self, h0, h1):
        l1 = self.semi_loss(h0, h1)

        return l1


class LogReg(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward(self, x):
        x = self.fc(x)

        return F.log_softmax(x, dim=1)