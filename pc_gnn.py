import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import dgl
import torch
from sklearn.metrics import accuracy_score
import numpy as np
from dgl.convert import from_networkx
import torch as th
import pandas as pd
import ast
from math import sqrt


class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='lstm')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='lstm')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h


def single_graph_data_process(G, W, v, P_max, label):
    G_nondiag = G - np.diag(np.diag(G))
    nx_graph = nx.from_numpy_matrix(G_nondiag)
    nfeat_self_weight = np.diag(G)
    nfeat_neibor_weight_in = np.sum(G_nondiag, axis=0)
    nfeat_neibor_weight_out = np.sum(G_nondiag, axis=1)
    graph_nfeature_arr = np.transpose(np.vstack((nfeat_self_weight, nfeat_neibor_weight_in, nfeat_neibor_weight_out, W, v, P_max)))
    # graph_nfeature_arr = np.transpose(np.vstack((nfeat_self_weight, nfeat_neibor_weight_in, nfeat_neibor_weight_out, W, v, P_max)))

    g_nfeature = th.tensor(graph_nfeature_arr, dtype=torch.float32)
    g_nlabel = th.tensor(label, dtype=torch.long)
    dgl_graph = from_networkx(nx_graph) 
    dgl_graph.ndata["feat"] = g_nfeature 
    dgl_graph.ndata["label"] = g_nlabel
    return dgl_graph


def model_test(trainset, testset):

    model = SAGE(6, 128, 2)
    opt = torch.optim.Adam(model.parameters())
    epoch_losses = []
    for epoch in range(10):
        epoch_loss = 0
        for train_graph in trainset:
            feats = train_graph.ndata['feat']
            # print("feats:", feats)
            # print("label:", train_graph.ndata['label'])
            logits = model(train_graph, feats)
            loss = F.cross_entropy(logits, train_graph.ndata['label'])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.detach().item()
            # epoch_loss /= (iter + 1)
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)

        model.eval()
        test_pred, test_label = [], []
        with torch.no_grad():
            for test_graph in testset:
                feats = test_graph.ndata['feat']
                pred = torch.softmax(model(test_graph, feats), 1)
                pred = torch.max(pred, 1)[1].view(-1)
                label = test_graph.ndata['label']

                test_pred += pred.detach().cpu().numpy().tolist()
                test_label += label.detach().cpu().numpy().tolist()
                # print("test_pred:", test_pred)
                # print("test_label:", test_label)
        print("accuracy: ", accuracy_score(test_label, test_pred))


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

    model_test(trainset, testset)
