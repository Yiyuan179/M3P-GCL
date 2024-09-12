import os.path as osp
from torch_geometric.datasets import Planetoid, Actor, WebKB
import torch_geometric.transforms as T

def load_data(name):
    path = osp.join(osp.expanduser('~'), 'datasets')
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        return Planetoid(path, name, transform=T.NormalizeFeatures())
    if name in ['Cornell', 'Texas', 'Wisconsin']:
        return WebKB(path, name, transform=T.NormalizeFeatures())
    if name in ['Actor']:
        path = osp.join(path, name)
        return Actor(path, transform=T.NormalizeFeatures())

if __name__ == '__main__':
    for name in ['Cora', 'CiteSeer', 'PubMed', 'Cornell', 'Texas', 'Wisconsin', 'Actor']:
        dataset = load_data(name)
        data = dataset[0]
        print(f'{name}: ', data)
