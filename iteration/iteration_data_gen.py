# with issues

import numpy as np
import pandas as pd
import csv
import random


def sca_sgl_cdst_3u(G,w,p_bar):

    p = np.array([[0.01, 0.15, 0.01]]).T
    tol = 10e-7
    err = 1
    while err > tol:
        p_temp = p
        alpha = G.dot(np.diag(p.T[0]))/(G.dot(p)+1)
        # print("alpha:",alpha)
        p = iteration_3u(G,w,p_bar, alpha)
        err = np.sum(np.abs(p_temp-p))
        # print("err:",err)
    print("p:", p)
    return p,alpha


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


def data_gen_v1():
    num = 1000
    sigma = np.array([[0.05, 0.05, 0.05]]).T

    with open('iteration_data.csv', 'w', newline='') as file:
        for i in range(num):
            print("i:", i)
            G_std = np.round(np.random.rand(3, 3) + np.diag(np.random.rand(3)) * 100, 2)

            G = G_std / 0.05
            p_bar = np.round(np.random.rand(3, 1) * 5, 2)
            print("p_bar:", p_bar)

            w = np.round(np.random.rand(3, 1), 2)

            p_star, alpha_star = sca_sgl_cdst_3u(G_std, w, p_bar)

            G_re = G_std.reshape((9, 1))
            rea1 = np.hstack((G_re.T, p_bar.T))
            print(rea1)
            alpha_star = alpha_star.reshape((9, 1))

            # 9+3+3+
            res = np.hstack((G_re.T, w.T, sigma.T, p_bar.T, alpha_star.T, p_star.T))  # 9+3+3
            mywriter = csv.writer(file, delimiter=',')
            mywriter.writerows(res)


def data_gen_v2():
    num = 5
    sigma = np.array([[0.05, 0.05, 0.05]]).T

    with open('iteration_data3.csv', 'w', newline='') as file:
        for i in range(num):
            print("i:", i)
            G_std = np.round(np.random.rand(3, 3) + np.diag(np.random.rand(3)) * 100, 2)

            G = G_std / 0.05
            p_bar = np.round(np.random.rand(3, 1) * 2, 2)
            print("p_bar:", p_bar)

            w = np.round(np.random.rand(3, 1), 2)

            p_star, alpha_star = sca_sgl_cdst_3u(G_std, w, p_bar)
            p_star_true = brutal_search(G_std, w, p_bar)
            print("=================")

            G_re = G_std.reshape((9, 1))
            rea1 = np.hstack((G_re.T, p_bar.T))
            alpha_star = alpha_star.reshape((9, 1))

            # 9+3+3+
            res = np.hstack((G_re.T, w.T, sigma.T, p_bar.T, alpha_star.T, p_star.T,p_star_true.T))  # 9+3+3
            mywriter = csv.writer(file, delimiter=',')
            mywriter.writerows(res)


if __name__ == '__main__':
    data_gen_v1()





