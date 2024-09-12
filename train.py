import argparse
from time import perf_counter as t
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import dropout_edge, homophily
from sklearn.metrics import f1_score
from model import *
from dataset import load_data
from utils import knn_graph

def train(args, model, x, main_edges, auxi_edges, optimizer):
    model.train()
    optimizer.zero_grad()
    main_edges = dropout_edge(main_edges, p=args.drop_edge_rate)[0]
    auxi_edges = dropout_edge(auxi_edges, p=args.drop_kg_edge_rate)[0]
    h0, h1 = model(x, main_edges, auxi_edges)
    loss = model.loss(h0, h1)
    loss.backward()
    optimizer.step()

    return loss.item()

def run(args, dataset, data, device, r):
    model = Model(dataset.num_features, args).to(device)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_p, weight_decay=args.wd_p)
    if args.ishomo == 1:
        pri_edges = data.edge_index
        sup_edges = data.kg_edge_index
    else:
        pri_edges = data.kg_edge_index
        sup_edges = data.edge_index
    start = t()
    prev = start
    cnt_wait = 0
    best = 1e9
    best_t = 0
    patience = 20
    for epoch in range(1, args.num_epochs + 1):
        loss = train(args, model, data.x, pri_edges, sup_edges, optimizer)
        now = t()
        if loss < best:
            best = loss
            cnt_wait = 0
            torch.save(model.state_dict(), 'model.pkl')
        else:
            cnt_wait += 1
        # print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now
        if cnt_wait == patience:
            # print(f'Early stopping at {epoch}!')
            break
    # print("===", r, "th run finished===")
    model.load_state_dict(torch.load('model.pkl'))
    model.eval()
    embeds = model(data.x, pri_edges, sup_edges, training=False)

    return embeds

def evaluate(model, embeds, data):
    model.eval()
    with torch.no_grad():
        logits = model(embeds)
    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        f1 = f1_score(data.y[mask].cpu().numpy(), pred.cpu().numpy(), average="macro")
        outs['{}_loss'.format(key)] = loss
        outs['{}_acc'.format(key)] = acc
        outs['{}_f1'.format(key)] = f1

    return outs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cornell')
    parser.add_argument('--SEED', type=int, default=0)
    parser.add_argument('--K', type=int, default=70)
    parser.add_argument('--pri_hop', type=int, default=2)
    parser.add_argument('--sup_hop', type=int, default=1)
    parser.add_argument('--global_hop', type=int, default=10)
    parser.add_argument('--lr_p', type=float, default=0.001)
    parser.add_argument('--lr_m', type=float, default=0.01)
    parser.add_argument('--wd_p', type=float, default=0.0001)
    parser.add_argument('--wd_m', type=float, default=0.0)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--num_hidden', type=int, default=256)
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--eps1', type=float, default=1.0)
    parser.add_argument('--eps2', type=float, default=1.0)
    parser.add_argument('--metric', type=str, default='cosine')
    parser.add_argument('--drop_edge_rate', type=float, default=0.1)
    parser.add_argument('--drop_kg_edge_rate', type=float, default=0.2)
    args = parser.parse_args()

    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed(args.SEED)
    np.random.seed(args.SEED)
    # torch.backends.cudnn.deterministic = True

    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    dataset = load_data(args.dataset)
    data = dataset[0]
    data.kg_edge_index = knn_graph(data.x, k=args.K, metric=args.metric)
    h = homophily(data.edge_index, data.y, method='edge')
    args.ishomo = 1 if h >= 0.5 else 0
    data = data.to(device)

    f_accs = []
    f_f1s = []
    for r in range(5):
        if len(data.train_mask.size()) > 1:
            data.train_mask = data.train_mask[:, r]
            data.val_mask = data.val_mask[:, r]
            data.test_mask = data.test_mask[:, r]
        embeds = run(args, dataset, data, device, r)
        train_lbls = data.y[data.train_mask]
        accs = []
        f1s = []
        for _ in range(10):
            log = LogReg(args.num_hidden, dataset.num_classes)
            opt = torch.optim.Adam(log.parameters(), lr=args.lr_m, weight_decay=args.wd_m)
            log.to(device)
            best_val_loss = float('inf')
            val_loss_history = []
            for _ in range(200):
                log.train()
                opt.zero_grad()
                logits = log(embeds[data.train_mask])
                loss = F.nll_loss(logits, train_lbls)
                loss.backward()
                opt.step()
                eval_info = evaluate(log, embeds, data)
                if eval_info['val_loss'] < best_val_loss:
                    best_val_loss = eval_info['val_loss']
                val_loss_history.append(eval_info['val_loss'])
                if _ > _ // 2:
                    tmp = torch.tensor(val_loss_history[-(20 + 1):-1])
                    if eval_info['val_loss'] > tmp.mean().item():
                        break
            accs.append(eval_info['test_acc'] * 100)
            f1s.append(eval_info['test_f1'] * 100)
        f_accs.append(np.mean(accs))
        f_f1s.append(np.mean(f1s))
    print(f"{args.dataset}: {np.mean(f_accs):.2f} {np.std(f_accs):.2f}, {np.mean(f_f1s):.2f} {np.std(f_f1s):.2f}")

if __name__ == '__main__':
    main()
