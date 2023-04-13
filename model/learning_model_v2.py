import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.datasets import fetch_california_housing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torch.utils.data as Data
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import random
import scipy
import csv

torch.random.seed()
def data_progress():
    cvxcnn_data = list()
    cvxcnn_target = list()
    with open("../data_generation/data.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            cvxcnn_data.append(list(map(float, line[:-3])))
            cvxcnn_target.append(list(map(float, line[-3:])))
            # print("cvxcnn_data:",cvxcnn_data)

    cvxcnn_data = np.array(cvxcnn_data)
    cvxcnn_target = np.array(cvxcnn_target)
    x_train,x_test,y_train,y_test = train_test_split(cvxcnn_data,cvxcnn_target,test_size=0.001,random_state=42)
    print("x_train:",x_train.shape,"x_test:",x_test.shape,"y_train:",y_train.shape,"y_test:",y_test.shape) #(14448, 8) (6192, 8) (14448,) (6192,)

    train_xt = torch.from_numpy(x_train.astype(np.float32))
    train_yt = torch.from_numpy(y_train.astype(np.float32))
    test_xt = torch.from_numpy(x_test.astype(np.float32))
    test_yt = torch.from_numpy(y_test.astype(np.float32))

    train_data = Data.TensorDataset(train_xt,train_yt)

    test_data = Data.TensorDataset(test_xt,test_yt)
    train_loader = Data.DataLoader(dataset=train_data,batch_size=1,shuffle=True,num_workers=0)
    print("train_loader:", train_loader)

    idx = random.randint(0,299)
    #idx = 249，132，272，160
    idx = 1
    print("idx:",idx)
    single_data = np.array([cvxcnn_data[idx].tolist()])
    single_target = cvxcnn_target[idx]
    # single_data = scale.transform(single_data)
    single_data = torch.from_numpy(single_data.astype(np.float32))
    return train_loader, train_xt, train_yt, test_xt, test_yt,y_test,single_data,single_target


class Cvxnnregression(nn.Module):
    def __init__(self):
        super(Cvxnnregression, self).__init__()
        self.neuron_num = 100
        self.hidden1 = nn.Linear(in_features=18,out_features=self.neuron_num,bias=True) #200
        # self.hidden2 = nn.Linear(self.neuron_num,self.neuron_num) #200*200
        # self.hidden3 = nn.Linear(self.neuron_num,self.neuron_num) #200*100
        # self.r_layer = nn.Linear(self.neuron_num,4)

    def forward(self,x):
        x0 = x
        # p_bar =  x0[0][15:18]
        p_bar = torch.reshape(x0[0][15:18], (3,1))

        p = p_bar
        #layer 1
        # x = self.single_condensationL(x0,p)
        x = x0[0][0:18]
        alpha = F.relu(nn.Linear(18, self.neuron_num)(x))
        alpha = F.relu(nn.Linear(self.neuron_num, 9)(alpha))
        p = self.cvxL_unroll(x0, alpha, p)
        # layer 2
        x = self.single_condensationL(x0, p)
        alpha = F.relu(nn.Linear(3, self.neuron_num)(x))
        alpha = F.relu(nn.Linear(self.neuron_num, 9)(alpha))
        p = self.cvxL_unroll(x0, alpha, p)
        # layer 3
        x = self.single_condensationL(x0, p)
        alpha = F.relu(nn.Linear(3, self.neuron_num)(x))
        alpha = F.relu(nn.Linear(self.neuron_num, 9)(alpha))
        p = self.cvxL_unroll(x0, alpha, p)
        return p.t()


    def single_condensationL(self,x0,p):
        G = torch.reshape(x0[0][0:9], (3,3)) / 0.05
        return torch.mm(G,p).t()


    def cvxL_unroll(self,x0,alpha,p_bar): # inital p: p_bar
        p = p_bar
        # layer 1
        p = self.unrollL(x0,alpha,p)
        # print("unrollLp:",p)
        p = p_bar-F.relu(p_bar-p)
        p = F.relu(p)

        # layer 2
        p = self.unrollL(x0, alpha, p)
        # print("unrollLp:",p)
        p = p_bar - F.relu(p_bar - p)
        p = F.relu(p)

        # layer 3
        p = self.unrollL(x0, alpha, p)
        # print("unrollLp:",p)
        p = p_bar - F.relu(p_bar - p)
        p = F.relu(p)

        return p


    def unrollL(self,x0,alpha,p):
        G = torch.reshape(x0[0][0:9], (3, 3)) / 0.05
        G = G - torch.diag(torch.diag(G))

        w = torch.reshape(x0[0][9:12], (3, 1))
        alpha = torch.reshape(alpha, (3,3))

        # print("p:",p)
        h = torch.mm(G, p) + 1
        hw = torch.cat((h, torch.mm(w.t(), G).t()), 0)
        m = nn.Linear(6, self.neuron_num)(hw.t())
        # print("m0:",m)
        m = nn.Linear(self.neuron_num, self.neuron_num)(m)

        m = nn.Linear(self.neuron_num, 3)(m).t()
        # print("m1:",m)

        p = torch.mm(w.t(),alpha).t() / m
        # print("unrollLp:",p)

        return p


def custom_mse(predicted, target):
    total_mse = 0

    # target = torch.tensor([[0.,0.,0.,0.]])  # add this line for unsupervised learning
    for i in range(target.shape[1]):
        # print("predicted[i]:", predicted.T[i])
        total_mse+=nn.MSELoss()(predicted.T[i], target.T[i])
    return total_mse


def training_progress(train_loader, train_xt, train_yt, test_xt, test_yt,y_test,single_data,single_target):
    epoch_num = 30
    cvxcnnreg = Cvxnnregression()
    optimizer = SGD(cvxcnnreg.parameters(),lr=0.0015,weight_decay=0.0001)
    loss_func = nn.MSELoss() 
    train_loss_all = []
    single_tp = []
    optimal_value = single_target.tolist()
    optimal_value = [optimal_value]*epoch_num
    # print("optimal_value:", optimal_value)

    for epoch in range(epoch_num):
        print("epoch:", epoch)
        train_loss = 0
        train_num = 0
        for step,(b_x,b_y) in enumerate(train_loader):
            # print("step:", step)
            p_predict = cvxcnnreg(b_x)

            # print("output:", output)
            # loss = loss_func(output,b_y)

            loss = custom_mse(p_predict, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_x.size(0)
            train_num += b_x.size(0)
        train_loss_all.append(train_loss / train_num)
        single_predict = cvxcnnreg(single_data)
        p  = single_predict.data.numpy()
        # print("====p:",p)
        # single_tp.append(np.sum(p))
        single_tp.append(p[0].tolist())
        print("train_loss_all:", train_loss / train_num)
    print("train_loss_all:",train_loss_all)
    return cvxcnnreg, single_tp,optimal_value



if __name__ == '__main__':
    run_time = 1
    obj_list = []
    train_loader, train_xt, train_yt, test_xt, test_yt, y_test, single_data, single_target = data_progress()
    if run_time == 1:
        cvxcnnreg, single_tp,optimal_value = training_progress(train_loader, train_xt, train_yt, test_xt, test_yt, y_test,single_data,single_target)
    if run_time > 1:
        for i in range(run_time):
            cvxcnnreg, single_tp,optimal_value = training_progress(train_loader, train_xt, train_yt, test_xt, test_yt,
                                                                    y_test, single_data, single_target)
            single_predict, r_predict = cvxcnnreg(single_data)
            p = single_predict.data.numpy()
            obj_list.append(np.sum(p[0]))

        print(obj_list)
        add_row = []
        with open('cvxL_res_2by2/obj_list_cnn.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ',
                                    quoting=csv.QUOTE_MINIMAL)

            spamwriter.writerow(obj_list)



