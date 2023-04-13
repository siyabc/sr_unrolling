import numpy as np
import pandas as pd
import csv
import random

# np.random.seed(1)


# alpha = np.array([[0.5,0.5],[0.5,0.5]])


def sca_sgl_cdst_2u(G):
    # G = np.array([[1, 2], [3, 4]])
    # p = np.array([[0.5],[0.5]])
    p = np.random.rand(2, 1)
    tol = 10e-7
    err = 1
    while err > tol:
        p_temp = p
        alpha = G.dot(np.diag(p.T[0]))/(G.dot(p)+1)
        print(alpha)
        p = iteration_2u(alpha)
        err = np.sum(np.abs(p_temp-p))
        print("err:",err)
        print("p:", p)


def iteration_2u(alpha):
    G_std = np.array([[0.6, 0.03], [0.08, 0.65]])
    G = G_std / 0.05
    p_bar = np.array([[1.5], [1.2]])
    # w = np.array([[1],[1]])
    w = np.random.rand(2, 1)

    p0 = 0.1
    p1 = 0.5
    tol = 10e-7
    err = 1

    while err>tol:
        p0_temp = p0
        p1_temp = p1
        # print("p0_temp:",p0_temp)

        p0 = min(w.T.dot(alpha[:,0])/(w[1]*G[1][0]/(G[1][0]*p0+1)),p_bar[0])
        p1 = min(w.T.dot(alpha[:,1])/(w[0]*G[0][1]/(G[0][1]*p1+1)),p_bar[1])
        # print("p0:",p0)
        # print("p1:",p1)

        err = abs(p0_temp - p0)+abs(p1_temp - p1)
        # print(err)

    return np.array([p0,p1])


def sca_sgl_cdst_3u(G,w,p_bar):

    p = np.array([[0.01, 0.15, 0.01]]).T
    tol = 10e-7
    err = 1
    while err > tol:
        p_temp = p
        alpha = G.dot(np.diag(p.T[0]))/(G.dot(p)+1)
        # print(alpha)
        p = iteration_3u(G,w,p_bar, alpha)
        err = np.sum(np.abs(p_temp-p))
        # print("err:",err)
    print("p:", p)
    return p


def iteration_3u(G,w,p_bar, alpha):
    p = np.random.rand(3, 1)
    p0 = p[0][0]
    p1 = p[1][0]
    p2 = p[2][0]

    tol = 10e-7
    err = 1

    while err>tol:

        p0_temp = p0
        p1_temp = p1
        p2_temp = p2
        d0 = w[1]*G[1][0]/(G[1][0]*p0+G[1][2]*p2+1) + w[2]*G[2][0]/(G[2][0]*p0+G[2][1]*p1+1)
        p0 = min(w.T.dot(alpha[:,0])/d0,p_bar[0])
        d1 = w[0]*G[0][1]/(G[0][1]*p1+G[0][2]*p2+1) + w[2]*G[2][1]/(G[2][0]*p0+G[2][1]*p1+1)
        p1 = min(w.T.dot(alpha[:,1])/d1,p_bar[1])
        d2 = w[0] * G[0][2] / (G[0][1] * p1 + G[0][2] * p2 + 1) + w[1] * G[1][2] / (G[1][0] * p0 + G[1][2] * p2 + 1)
        p2 = min(w.T.dot(alpha[:, 2]) / d2, p_bar[2])
        # print("p0:",p0)
        # print("p1:",p1)
        # print("p2:",p2)

        err = abs(p0_temp - p0)+abs(p1_temp - p1)+abs(p2_temp - p2)
        # print(err)

    return np.array([p0,p1,p2])


def brutal_search(G,w,p_bar):
    diff = 0.01
    sigma = np.array([[0.05, 0.05, 0.05]]).T
    max_obj = 10e-8
    F = np.dot(np.linalg.inv(np.diag(np.diag(G))), (G - np.diag(np.diag(G))))
    v = np.dot(np.linalg.inv(np.diag(np.diag(G))), sigma)
    for p_0 in np.arange(10e-7, p_bar[0][0], diff):
        for p_1 in np.arange(10e-7, p_bar[1][0], diff):
            for p_2 in np.arange(10e-7, p_bar[2][0], diff):
                p = np.array([[p_0],[p_1],[p_2]])

                sinr = (1 / (np.dot(F, p) + v)) * p # 2*1,m=1
                f_func = np.log(1 + sinr)
                obj = w.T.dot(f_func)[0][0]
                # print("obj:", obj)

                if obj>=max_obj:
                    max_obj = obj
                    # print("max_obj:", max_obj)
                    # print("pinter:", p)
                    p_star = p
    print("p_star:", p_star)
    return p_star

if __name__ == '__main__':
    # iteration_2u(alpha)
    # sca_sgl_cdst()
    # alpha = np.array([[0.3, 0.3, 0, 3], [0.3, 0.3, 0, 3], [0.3, 0.3, 0, 3]])

    # iteration_3u(alpha)

    G_std = np.array([[0.4, 0.03, 0.02], [0.08, 0.65, 0.02], [0.08, 0.05, 0.52]])
    G = G_std / 0.05
    p_bar = np.array([[1.5], [1.2], [1.6]])
    w = np.array([[0.5], [0.2], [0.6]])

    # single_condensation
    flag = False
    p = sca_sgl_cdst_3u(G,w,p_bar)
    p_temp = p
    while flag:
        p = sca_sgl_cdst_3u(G,w,p_bar)
        if abs(np.sum(p) - np.sum(p_temp))>10e-2:
            flag = False
            print("p_temp:",p_temp)
            print("p:",p)

    #brutal search
    brutal_search(G_std, w, p_bar)




# p: [[1.5       ]
#  [0.45323168]
#  [1.6       ]]






