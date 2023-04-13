import numpy as np
import matplotlib.pyplot as plt


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
        err = abs(p0_temp - p0)+abs(p1_temp - p1)
    return np.array([p0,p1])


def sca_sgl_cdst_3u(G,w,p_bar):
    p_list = []
    obj_list = []
    p = np.array([[0.01, 0.15, 0.01]]).T
    tol = 10e-9
    err = 1
    while err > tol:
        p_temp = p
        alpha = G.dot(np.diag(p.T[0]))/(G.dot(p)+1)
        p,p_iter_list,obj_iter_list = iteration_3u(G,w,p_bar, alpha,p)
        err = np.sum(np.abs(p_temp-p))
        p_list = p_list+ p_iter_list
        obj_list = obj_list+obj_iter_list
        # p_list.append(p.T[0].tolist())
        print("err:",err)
    print("p:", p)
    print("alpha:", alpha)

    return p,p_list,obj_list


def iteration_3u(G,w,p_bar, alpha,p):
    # p = np.random.rand(3, 1)
    p0 = p[0][0]
    p1 = p[1][0]
    p2 = p[2][0]

    tol = 10e-7
    err = 1

    p_list = []
    obj_list = []
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
        err = abs(p0_temp - p0)+abs(p1_temp - p1)+abs(p2_temp - p2)
        p_list.append([p0[0],p1[0],p2[0]])
        obj = obj_func(G, w, np.array([p0,p1,p2]))
        obj_list.append(obj)
    return np.array([p0,p1,p2]),p_list,obj_list


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


def obj_func(G,w,p):
    sigma = np.array([[0.05, 0.05, 0.05]]).T
    F = np.dot(np.linalg.inv(np.diag(np.diag(G))), (G - np.diag(np.diag(G))))
    v = np.dot(np.linalg.inv(np.diag(np.diag(G))), sigma)
    sinr = (1 / (np.dot(F, p) + v)) * p  # 2*1,m=1
    f_func = np.log(1 + sinr)
    obj = w.T.dot(f_func)[0][0]
    return obj


def plot_power(single_tp,optimal_value):
    single_tp = np.array(single_tp)
    optimal_value = np.array(optimal_value)
    # plot loss
    plt.figure(figsize=(8,6))
    color_choice =  ['red','blue','green','purple']

    for i in range(3):
        print("i:",i)
        plt.plot(single_tp[:,i], label="User "+str(i+1)+  "(Algorithm 1)", color = color_choice[i], alpha=0.7)
        plt.plot(optimal_value[:,i], linewidth=2, linestyle="-." , label="User "+str(i+1)+  "(Ground truth)",color = color_choice[i],alpha=0.7)
    num1 = 1.01
    num2 = 0
    num3 = 3
    num4 = 0
    # plt.legend(fontsize=14, bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)
    plt.legend(fontsize=14)
    plt.xlabel("iterations", fontsize=14)
    plt.ylabel("power(W)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig("bad_sca.pdf")
    plt.show()



def plot_obj(obj,optimal_obj):

    # plot loss
    plt.figure(figsize=(8,6))
    color_choice =  ['red','blue','green','purple']


    plt.plot(optimal_obj, linewidth=2, linestyle="-" , label="Ground truth",color = color_choice[0],alpha=0.7)
    plt.plot(obj, label="Algorithm 1", color = color_choice[1], alpha=0.7)

    num1 = 1.01
    num2 = 0
    num3 = 3
    num4 = 0
    # plt.legend(fontsize=14, bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)
    plt.legend(fontsize=14)
    plt.xlabel("iterations", fontsize=14)
    plt.ylabel("Sum-rate", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig("bad_sca_obj.pdf")
    plt.show()


if __name__ == '__main__':

    G_std = np.array( [[32.18,  0.24 , 0.97],
 [ 0.53, 58.87,  0.84],
 [ 0.25 , 0.81 ,33.32]])
 #    G_std = np.array([[50.18 , 0.49 , 0.68],
 #     [0.63, 98.64,  0.34],
 #    [0.19,0.71,7.92]])
    print("G_std:",G_std)

    G = G_std / 0.05
    p_bar = np.array([[2.5], [4], [5]])
    w = np.array([[0.5], [0.2], [0.6]])

    # single_condensation
    p,p_list,obj_list = sca_sgl_cdst_3u(G,w,p_bar)
    obj = obj_func(G, w, p)
    print("obj:",obj)
    print("p_list:",len(p_list))

    # test
    p_search = np.array([[9.000100e-02],[1.000000e-06],[4.990001e+00]])
    # p_search = np.array([[2.490001],[0.270001],[4.990001]])
    opt_p = [p_search.T[0].tolist()]*len(p_list)
    print("opt_p:",opt_p)

    opt_obj = obj_func(G, w, p_search)
    print("obj3:", opt_obj)
    plot_obj(obj_list, [opt_obj]*len(p_list))

    plot_power(p_list, opt_p)








