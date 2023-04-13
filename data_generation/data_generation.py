import numpy as np
import pandas as pd
import csv
import random


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
    num = 50
    sigma = np.array([[0.05, 0.05, 0.05]]).T

    with open('data4.csv', 'w', newline='') as file:
        for i in range(num):
            print("i:", i)
            G_std = np.round(np.random.rand(3, 3) + np.diag(np.random.rand(3)) * 100,2)

            G = G_std / 0.05
            p_bar = np.round(np.random.rand(3, 1)*5, 2)
            print("p_bar:",p_bar)

            w = np.round(np.random.rand(3, 1), 2)

            p_star = brutal_search(G_std, w, p_bar)

            G_re = G_std.reshape((9,1))
            rea1 = np.hstack((G_re.T, p_bar.T))
            print(rea1)

            res = np.hstack((G_re.T,w.T,sigma.T,p_bar.T,p_star.T)) # 9+3+3
            mywriter = csv.writer(file, delimiter=',')
            mywriter.writerows(res)


