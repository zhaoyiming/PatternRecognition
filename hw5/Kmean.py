import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


data = [[-7.82, -6.68, 4.36, 6.72, -8.64, -6.87, 4.47, 6.73, -7.71, -6.91, 6.18, 6.72, -6.25, -6.94, 8.09, 6.81, -5.19,
         -6.38, 4.08, 6.27],
        [-4.58, 3.16, -2.19, 0.88, 3.06, 0.57, -2.62, -2.01, 2.34, -0.49, 2.81, -0.93, -0.26, -1.22, 0.20, 0.17, 4.24,
         -1.74, 1.30, 0.93],
        [-3.97, 2.71, 2.09, 2.80, 3.50, -5.45, 5.76, 4.18, -6.33, -5.68, 5.82, -4.04, 0.56, 1.13, 2.25, -4.15, 4.04,
         1.43, 5.33, -2.78]]
W = np.array(data).T
center1 = np.array([[1.0, 1.0, 1.0], [-1.0, 1.0, -1.0]])
center2 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, -1.0]])
center3 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [-1.0, 0.0, 2.0]])
center4 = np.array([[-0.1, 0.0, 0.1], [0.0, -0.1, 0.1], [-0.1, -0.1, 0.1]])

def plotit(array, center, cl, mode, theta):
    fig = plt.figure()
    ax2 = Axes3D(fig)

    for i in range(array.shape[0]):
        for j in array[i]:
            if i == 0:
                ax2.scatter3D(j[0], j[1], j[2], color='red', s=10)
                ax2.scatter3D(center[i][0], center[i][1], center[i][2], color='red', marker="s", s=30)
            elif i == 1:
                ax2.scatter3D(j[0], j[1], j[2], color='blue', s=10)
                ax2.scatter3D(center[i][0], center[i][1], center[i][2], color='blue', marker="s", s=30)
            elif i == 2:
                ax2.scatter3D(j[0], j[1], j[2], color='green', s=10)
                ax2.scatter3D(center[i][0], center[i][1], center[i][2], color='green', marker="s", s=30)
    if mode == 0:
        plt.savefig('Kmeans/Kmeans-euc' + str(cl) + '.png', bbox_inches='tight')
    elif mode == 1:
        plt.savefig('Kmeans/Kmeans-custom' + str(cl) + '+' + str(theta) + '.png', bbox_inches='tight')
    plt.show()

def mdis(x, y, theta):
    res = np.sqrt(np.sum(np.power((x - y), 2)))
    mres= 1-np.exp(-theta * np.power(res,2))
    return mres
def dis(x,y):
    res = np.sqrt(np.sum(np.power((x-y), 2)))
    return res


def distance(data, center, mode, theta):
    l= data.shape[0]
    m= center.shape[0]
    distance= np.zeros((l, m))
    for i in range(m):
        for j in range(l):
            if mode == 0:
                distance[j][i] = dis(center[i], data[j])
            elif mode == 1:
                distance[j][i] = mdis(center[i], data[j], theta)
    return distance


def kmean(w, center, cl, mode):
    theta=1
    lsize = center.shape[0]
    ind=0
    while True:
        ind= ind+1
        temp = center.copy()
        points = [[] for key in range(lsize)]
        mdistance = distance(w, center, mode, theta)
        for p in range(mdistance.shape[0]):
            nearest_index = np.argmin(mdistance[p])
            points[nearest_index].append(w[p])
        points_final = np.asarray(points)
        for i in range(lsize):
            center[i] = np.sum(points_final[i], axis=0) / len(points_final[i])

        if (temp == center).all():
            break
        if ind>100:
            break
    if mode==0:
        print("K-均值算法，第", cl, "个初始矩阵, 距离指标为欧氏距离")
    elif mode==1:
        print("K-均值算法，第", cl, "个初始矩阵, 距离指标为自定义，beta值为", theta)
    print("聚类点：")
    print(center)
    print("------------------------------")
    for i in range(len(points_final)):
        print("第", i, "类点-----------------")
        for j in points_final[i]:
            print(j)
    print("循环次数：", ind)
    plotit(points_final, center, cl, mode, theta)


if __name__ == '__main__':
    # kmean(W, center1, 1, 1)  # 分别表示数据、聚类中心、初始聚类序号、距离公式类别
    # kmean(W, center2, 2, 1)
    # kmean(W, center3, 3, 1)
    kmean(W, center4, 4, 1)
