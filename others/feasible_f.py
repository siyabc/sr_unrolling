import numpy as np
import csv


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

def try_f():
    i = 0
    while 1:
        print("i=", i)
        G = np.random.rand(3, 3)
        G = G - np.diag(np.diag(G))
        p_bar = np.round(np.random.rand(3, 1) * 5, 2)
        w = np.random.rand(3, 1)
        alpha = np.random.rand(3, 3)
        p = iteration_3u(G, w, p_bar, alpha)
        i = i+1


def iteration_FP():
    G = np.random.rand(3, 3)
    G = G - np.diag(np.diag(G))

# def try_f_v2():





