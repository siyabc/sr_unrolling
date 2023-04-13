import numpy as np
import cvxpy as cp
from math import log
import pandas as pd
import ast


def sinr_upperbound_cvx_solve(F, user_num, delete_node, v, P_max):
    F_temp = np.delete(F, delete_node, axis=0)
    F_temp = np.delete(F_temp, delete_node, axis=1)
    v_temp = np.delete(v, delete_node, axis=1)
    P_max_temp = np.delete(P_max, delete_node)
    v_delete = v[:][:, delete_node]
    P_max_delete = P_max[:][:, delete_node]

    remain_len = user_num - len(delete_node)
    B = np.identity(remain_len) + F_temp + np.sum(P_max_temp) * np.matmul(v_temp.T, np.ones((1, remain_len)))
    B_inv = np.linalg.inv(B)
    C = np.eye(remain_len) - B_inv
    # print("B1:", B)
    y = cp.Variable(shape=remain_len, pos=True)
    obj = cp.prod(C @ y)
    cons = [y >= np.dot(B, P_max_temp.T)]
    problem = cp.Problem(cp.Minimize(obj), cons)
    problem.solve(gp=True)
    # print("y1:", y.value)

    p = np.atleast_2d(B_inv @ y.value).T
    p = p*(np.sum(P_max_temp)/np.sum(p))
    # print("p1:", p)
    # print("p1 sum:", np.sum(p))
    gamma = np.multiply(p, 1/(np.dot(F_temp, p)+v))
    R_tot = np.log(np.prod(gamma))

    gamma_del = np.multiply(P_max_delete, 1/v_delete) + 1
    R_tot_del = np.log(np.prod(gamma_del))

    R_tot_sum = R_tot + R_tot_del
    return R_tot_sum


def sinr_upperbound_cvx_solve2(F, user_num, delete_node, v, P_max):
    F_temp = np.delete(F, delete_node, axis=0)
    F_temp = np.delete(F_temp, delete_node, axis=1)
    v_temp = np.delete(v, delete_node, axis=1)
    P_max_temp = np.delete(P_max, delete_node)
    v_delete = v[:][:, delete_node]
    P_max_delete = P_max[:][:, delete_node]

    remain_len = user_num - len(delete_node)
    B = np.identity(remain_len) + F_temp + np.sum(P_max_temp) * np.matmul(v_temp.T, np.ones((1, remain_len)))
    B_inv = np.linalg.inv(B)
    C = np.eye(remain_len) - B_inv

    y = cp.Variable(shape=remain_len, pos=True)
    obj = cp.prod(C @ y)
    # cons = [y >= np.dot(B, P_max_temp.T)]
    cons = [y >= C@y, cp.prod(y) == 1]

    problem = cp.Problem(cp.Minimize(obj), cons)
    problem.solve(gp=True)
    # print("y2:", y.value)

    p = np.atleast_2d(B_inv @ y.value).T
    p = p*(np.sum(P_max_temp)/np.sum(p))
    # print("p2 sum:", np.sum(p))
    gamma = np.multiply(p, 1/(np.dot(F_temp, p)+v))+1
    R_tot = np.sum(np.log(gamma))

    gamma_del = np.multiply(P_max_delete, 1/v_delete) + 1
    R_tot_del = np.sum(np.log(gamma_del))

    R_tot_sum = R_tot + R_tot_del
    return R_tot_sum


def sinr_upperbound_cvx_solve3(B, W_temp, W_delete, v_delete, P_max_delete):
    a, b = np.linalg.eig(B)
    idx = np.argmax(a, axis=0)
    aT, bT = np.linalg.eig(B.T)
    idxT = np.argmax(a, axis=0)
    rtot = np.max(np.multiply(W_temp, 1/np.multiply(b[idx], bT[idxT])) * log(np.max(a)))
    upper_sumrate = rtot

    if len(P_max_delete) > 0:
        upper_sumrate = rtot + np.sum(np.multiply(W_delete, np.log(1+np.multiply(P_max_delete, 1/v_delete))))
    return upper_sumrate


def find_B_tilta(F, W, v, P_max, user_num):
    delete_list = [[i] for i in list(range(user_num))]
    delete_list_copy = delete_list.copy()
    R_tot_dict = dict()
    for delete_node in delete_list:
        F_temp = F.copy()
        W_temp = W.copy()
        v_temp = v.copy()
        P_max_temp = P_max.copy()
        user_num_temp = user_num
        node_list = list(range(user_num))
        remian_node = list(set(node_list).difference(set(delete_node)))

        F_temp = np.delete(F_temp, delete_node, axis=0)
        F_temp = np.delete(F_temp, delete_node, axis=1)
        W_temp = np.delete(W_temp, delete_node, axis=1)
        v_temp = np.delete(v_temp, delete_node, axis=1)
        P_max_temp = np.delete(P_max_temp, delete_node)
        user_num_temp = user_num_temp - len(delete_node)
        W_delete = W[:][:, delete_node]
        v_delete = v[:][:, delete_node]
        P_max_delete = P_max[:][:, delete_node]

        is_B_tilta, B = valid_B(F_temp, v_temp, P_max_temp, user_num_temp)
        if is_B_tilta:
            delete_list_copy.remove(delete_node)
            R_tot = sinr_upperbound_cvx_solve3(B, W_temp, W_delete, v_delete, P_max_delete)
            R_tot_dict[str(remian_node)] = R_tot
        else:
            delete_list_copy.remove(delete_node)
            if max(delete_node) < user_num - 1:
                for i in range(max(delete_node)+1, user_num):
                    delete_list.append(delete_node + [i])
                    delete_list_copy.append(delete_node + [i])
    return min(R_tot_dict, key=R_tot_dict.get)
    # return min(R_tot_dict.values())


def valid_B(F, v, P_max, user_num):
    # B = F + np.sum(P_max) * np.matmul(v.T, np.ones((1, user_num)))
    # B_tilta = np.dot(np.linalg.inv(np.identity(user_num) + B), B)

    B = np.identity(user_num) + F + np.sum(P_max) * np.matmul(v.T, np.ones((1, user_num)))
    B_inv = np.linalg.inv(B)
    B_tilta = np.eye(user_num) - B_inv

    is_B_tilta = (B_tilta >= 0).all()
    return is_B_tilta, B


def trainset_ml():
    all_num = 0
    G_self_list = []
    G_in_list = []
    G_out_list = []
    W_list = []
    v_list = []
    P_max_list = []
    label_list = []
    for i in range(500):
        print("i:", i)
        # user_num = np.random.randint(2, 15, size=1)[0]
        user_num = 5
        print("usernum:", user_num)
        diag_m = 200 * np.random.rand(user_num, user_num)
        G = 10 * np.random.rand(user_num, user_num) + np.diag(np.diag(diag_m))
        P_max = 10 * np.random.rand(1, user_num)
        W = 5 * np.random.rand(1, user_num)
        v = 2 * np.random.rand(1, user_num)

        F = G / (np.diag(G)) - np.identity(user_num)
        v = np.multiply(v, 1 / (np.diag(G)))
        is_B_tilta, B_init = valid_B(F, v, P_max, user_num)

        if is_B_tilta:
            B_in_F = str(list(range(user_num)))
            all_num = all_num + 1
            # print("**all B_in_F:", B_in_F)
        else:
            B_in_F = find_B_tilta(F, W, v, P_max, user_num)
        # W_list.append(list(W.ravel()))
        # v_list.append(list(v.ravel()))
        # P_max_list.append(list(P_max.ravel()))
        label = np.zeros(user_num)
        label[ast.literal_eval(B_in_F)] = 1
        # label_list.append(list(label))

        G_self = np.diag(G)
        G_nondiag = G - np.diag(np.diag(G))
        G_in = np.sum(G_nondiag, axis=0)
        G_out = np.sum(G_nondiag, axis=1)

        for i in range(user_num):
            G_self_list.append(G_self[i])
            G_in_list.append(G_in[i])
            G_out_list.append(G_out[i])
            W_list.append(W.ravel()[i])
            v_list.append(v.ravel()[i])
            P_max_list.append(P_max.ravel()[i])
            label_list.append(label[i])
    print("all_num:", all_num)

    dataframe = pd.DataFrame({'G_self': G_self_list, 'G_in': G_in_list, 'G_out': G_out_list, 'W_list': W_list, 'v': v_list, 'P_max': P_max_list, 'label': label_list})
    dataframe.to_csv("traingset_ml.csv", index=False, sep=',')


def trainset_gnn():
    all_num = 0
    G_list = []
    W_list = []
    v_list = []
    P_max_list = []
    label_list = []
    for i in range(500):
        print("i:", i)
        user_num = np.random.randint(2, 15, size=1)[0]
        # user_num = 5
        print("usernum:", user_num)
        diag_m = 500 * np.random.rand(user_num, user_num)
        G = 10 * np.random.rand(user_num, user_num) + np.diag(np.diag(diag_m))
        P_max = 10 * np.random.rand(1, user_num)
        W = 5 * np.random.rand(1, user_num)
        v = 2 * np.random.rand(1, user_num)
        # G = np.array([[6,2,1,3,1], [1,4,1,2,1], [1,2,5,3,1], [3,2,1,8,1], [1,1,2,2,6]])
        # P_max = np.array([[3,5,4,13,6]])
        # W = np.array([[1,1,1,1,1]])
        # v = np.array([[0.5, 0.5, 0.5, 0.5, 0.5]])

        F = G / (np.diag(G)) - np.identity(user_num)
        v = np.multiply(v, 1/(np.diag(G)))
        is_B_tilta, B_init = valid_B(F, v, P_max, user_num)

        if is_B_tilta:
            B_in_F = str(list(range(user_num)))
            all_num = all_num + 1
            # print("**all B_in_F:", B_in_F)
        else:
            B_in_F = find_B_tilta(F, W, v, P_max, user_num)
        G_list.append(str(list(G.ravel())))
        W_list.append(str(list(W.ravel())))
        v_list.append(str(list(v.ravel())))
        P_max_list.append(str(list(P_max.ravel())))
        label = np.zeros(user_num)
        print("B_in_F:", B_in_F)

        label[ast.literal_eval(B_in_F)] = 1
        label_list.append(str(list(label)))
    print("all_num:", all_num)

    dataframe = pd.DataFrame({'G': G_list, 'W': W_list, 'v': v_list, 'P_max': P_max_list, 'label': label_list})
    dataframe.to_csv("traingset.csv", index=False, sep=',')


if __name__ == '__main__':
    trainset_gnn()
    # trainset_ml()



