"""
.. _model-gcn:

Graph Convolutional Network

"""

import numpy as np
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import torch
from sklearn.metrics import accuracy_score
from dgl.convert import from_networkx
import pandas as pd
import ast
from math import sqrt


gcn_msg = fn.copy_u(u='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = GCNLayer(6, 128)
        self.layer2 = GCNLayer(128, 2)
    
    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        return x


def single_graph_data_process(G, W, v, P_max, label):
    G_nondiag = G - np.diag(np.diag(G))
    nx_graph = nx.from_numpy_matrix(G_nondiag)
    nfeat_self_weight = np.diag(G)
    nfeat_neibor_weight_in = np.sum(G_nondiag, axis=0)
    nfeat_neibor_weight_out = np.sum(G_nondiag, axis=1)
    # graph_nfeature_arr = np.transpose(np.vstack((nfeat_self_weight,nfeat_neibor_weight_in,nfeat_neibor_weight_out, W, v, P_max)))
    graph_nfeature_arr = np.transpose(
        np.vstack((nfeat_self_weight, nfeat_neibor_weight_in, nfeat_neibor_weight_out, W, v, P_max)))
    g_nfeature = th.tensor(graph_nfeature_arr, dtype=torch.float32)
    g_nlabel = th.tensor(label, dtype=torch.long)
    dgl_graph = from_networkx(nx_graph) 
    dgl_graph.ndata["feat"] = g_nfeature
    dgl_graph.ndata["label"] = g_nlabel
    return dgl_graph


def model_train(trainset, testset):
    net = Net()
    print(net)
    # g, features, labels, train_mask, test_mask = load_cora_data()
    # Add edges between each node and itself to preserve old node representations
    # g.add_edges(g.nodes(), g.nodes())
    optimizer = th.optim.Adam(net.parameters(), lr=1e-2)
    dur = []
    net.train()
    for epoch in range(15):

        for train_graph in trainset:
            train_graph.add_edges(train_graph.nodes(), train_graph.nodes())
            features = train_graph.ndata['feat']
            labels = train_graph.ndata['label']

            logits = net(train_graph, features)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        net.eval()
        test_pred, test_label = [], []
        with torch.no_grad():
            for test_graph in testset:
                feats = test_graph.ndata['feat']
                pred = torch.softmax(net(test_graph, feats), 1)
                pred = torch.max(pred, 1)[1].view(-1)
                label = test_graph.ndata['label']

                test_pred += pred.detach().cpu().numpy().tolist()
                test_label += label.detach().cpu().numpy().tolist()

        print("accuracy: ", accuracy_score(test_label, test_pred))
        # acc = evaluate(net, g, features, labels, test_mask)
        # print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
        #         epoch, loss.item(), acc, np.mean(dur)))


if __name__ == '__main__':
    df = pd.read_csv('traingset.csv')
    trainset = []
    testset = []
    for idx, row in df.iterrows():
        # print("G_row:", row["G"])
        G = ast.literal_eval(row["G"])
        user_num = int(sqrt(len(G)))
        G = np.reshape(np.array(G), (user_num, user_num))
        W = ast.literal_eval(row["W"])
        v = ast.literal_eval(row["v"])
        P_max = ast.literal_eval(row["P_max"])
        label = ast.literal_eval(row["label"])
        single_dgl_graph = single_graph_data_process(G, W, v, P_max, label)
        # print(single_dgl_graph)
        if idx % 2 != 0:
            trainset.append(single_dgl_graph)
        else:
            testset.append(single_dgl_graph)

    # model_test(trainset, testset)
    model_train(trainset, testset)