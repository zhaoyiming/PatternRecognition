import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import linalg
import math

np.set_printoptions(precision=4)

w2 = np.mat([
    [-0.4, -0.31, 0.38, -0.15, -0.35, 0.17, -0.011, -0.27, -0.065, -0.12],
    [0.58, 0.27, 0.055, 0.53, 0.47, 0.69, 0.55, 0.61, 0.49, 0.054],
    [0.089, -0.04, -0.035, 0.011, 0.034, 0.1, -0.18, 0.12, 0.0012, -0.063]
]).T

w3 = np.mat([[0.83, 1.1, -0.44, 0.047, 0.28, -0.39, 0.34, -0.3, 1.1, 0.18],
             [1.6, 1.6, -0.41, -0.45, 0.35, -0.48, -0.079, -0.22, 1.2, -0.11],
             [-0.014, 0.48, 0.32, 1.4, 3.1, 0.11, 0.14, 2.2, -0.46, -0.49]
             ]).T


def lda(w1, w2):
    w1_mean = np.mean(w1, axis=0)
    w2_mean = np.mean(w2, axis=0)

    var1 = np.dot((w1 - w1_mean).T, (w1 - w1_mean))
    var2 = np.dot((w2 - w2_mean).T, (w2 - w2_mean))
    var_all = var1 + var2
    var_inverse = var_all.I
    w = np.dot(var_inverse, (w1_mean - w2_mean).T)
    print(w)
    return w


def plotc(w2, w3):
    w = lda(w2, w3)
    w = np.array(w)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 绘制空间点
    ax.scatter(w2[:, 0], w2[:, 1], w2[:, 2], c='r', s=5)
    ax.scatter(w3[:, 0], w3[:, 1], w3[:, 2], c='b', s=5)
    # 画投影方向
    xline = np.array([-1, 1])
    yline = xline * w[1] / w[0]
    zline = xline * w[2] / w[0]
    ax.plot(xline, yline, zline, linewidth=3)

    wor = linalg.orth(w)
    w2s = np.array((wor * (wor.T * w2.T)).T)
    w3s = np.array((wor * (wor.T * w3.T)).T)
    # 投影到直线上的点
    ax.scatter(w2s[:, 0], w2s[:, 1], w2s[:, 2], c='r', s=5)
    ax.scatter(w3s[:, 0], w3s[:, 1], w3s[:, 2], c='b', s=5)
    w2 = np.array(w2)
    w3 = np.array(w3)
    # 连接线
    for i in range(10):
        ax.plot([w2[i][0], w2s[i][0]], [w2[i][1], w2s[i][1]], [w2[i][2], w2s[i][2]], "b--", linewidth=1)  # 画样本点与投影点的连线
        ax.plot([w3[i][0], w3s[i][0]], [w3[i][1], w3s[i][1]], [w3[i][2], w3s[i][2]], "k--", linewidth=1)
    plt.savefig("lda.jpg", dpi=500)
    plt.show()


def MSE():
    a = 0
    print(a)


def plotmse(w2, w3):
    w = lda(w2, w3)
    w = np.array(w)
    w21 = np.array(w2 * w)
    w31 = np.array(w3 * w)
    plt.scatter(w21, np.zeros([10, 1]), label="w2")  # 画样本点
    plt.scatter(w31, np.zeros([10, 1]), label="w3")
    m21, m31 = np.mean(w21, keepdims=True), np.mean(w31, keepdims=True)
    s21, s31 = np.var(w21), np.var(w31)

    t = np.expand_dims(np.linspace(-0.5, 0.5, 100), axis=1)
    p2 = np.exp(-np.power((t - m21), 2) / (2 * s21)) / np.sqrt(2 * math.pi * s21)
    p3 = np.exp(-np.power((t - m31), 2) / (2 * s31)) / np.sqrt(2 * math.pi * s31)
    plt.plot(t, p2, label="p(x|w2)")  # 画pdf曲线
    plt.plot(t, p3, label="p(x|w3)")

    f2 = s31 - s21
    f1 = -2 * (s31 * m21 - s21 * m31)
    f0 = s31 * np.power(m21, 2) - s21 * np.power(m31, 2) - (s21 * s31 * np.log(s31 / s21))
    f = [f2, f1, f0]
    root = np.roots(f)

    res = 0
    for i in range(len(root)):
        if max(m21, m31) > root[i] > min(m21, m31):
            res = root[i]
    print("分类决策面:x=", res)
    c = 0
    for i in range(len(w21)):
        if w21[i] < res:
            c += 1
        if w31[i] > res:
            c += 1
    error = c / (len(w21) + len(w31))
    print("训练误差:", error)
    plt.axvline(res, linestyle='--', c="k", label="bound")
    plt.savefig("plotbest.jpg")
    plt.show()



def plotworse(w2, w3):
    w = np.mat([1.0, 2.0, -1.5]).T
    w = np.array(w)
    w21 = np.array(w2 * w)
    w31 = np.array(w3 * w)
    plt.scatter(w21, np.zeros([10, 1]), label="w2")  # 画样本点
    plt.scatter(w31, np.zeros([10, 1]), label="w3")
    m21, m31 = np.mean(w21, keepdims=True), np.mean(w31, keepdims=True)
    s21, s31 = np.var(w21), np.var(w31)

    t = np.expand_dims(np.linspace(-5, 5, 100), axis=1)
    p2 = np.exp(-np.power((t - m21), 2) / (2 * s21)) / np.sqrt(2 * math.pi * s21)
    p3 = np.exp(-np.power((t - m31), 2) / (2 * s31)) / np.sqrt(2 * math.pi * s31)
    plt.plot(t, p2, label="p(x|w2)")
    plt.plot(t, p3, label="p(x|w3)")

    f2 = s31 - s21
    f1 = -2 * (s31 * m21 - s21 * m31)
    f0 = s31 * np.power(m21, 2) - s21 * np.power(m31, 2) - (s21 * s31 * np.log(s31 / s21))
    f = [f2, f1, f0]
    root = np.roots(f)

    res = 0
    for i in range(len(root)):
        if max(m21, m31) > root[i] > min(m21, m31):
            res = root[i]
    print("分类决策面:x=", res)
    c = 0
    for i in range(len(w21)):
        if w21[i] < res:
            c += 1
        if w31[i] > res:
            c += 1
    error = c / (len(w21) + len(w31))
    print("训练误差:", error)
    plt.axvline(res, linestyle='--', c="k", label="bound")
    plt.savefig("plotanother.jpg")
    plt.show()

if __name__ == '__main__':
    # plotc(w2, w3)
    plotmse(w2, w3)
    plotworse(w2, w3)
