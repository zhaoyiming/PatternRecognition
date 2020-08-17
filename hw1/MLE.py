import numpy as np
from scipy.optimize import minimize
import scipy.stats as stats
import time
import math

np.set_printoptions(precision=4)
x1 = np.array([0.42, -0.2, 1.3, 0.39, -1.6, -0.029, -0.23, 0.27, -1.9, 0.87])
x2 = np.array([-0.087, -3.3, -0.32, 0.71, -5.3, 0.89, 1.9, -0.3, 0.76, -1.0])
x3 = np.array([0.58, -3.4, 1.7, 0.23, -0.15, -4.7, 2.2, -0.87, -2.1, -2.6])
x = np.stack((x1, x2, x3), axis=0).T

w = np.array([
    [-0.4, -0.31, 0.38, -0.15, -0.35, 0.17, -0.011, -0.27, -0.065, -0.12],
    [0.58, 0.27, 0.055, 0.53, 0.47, 0.69, 0.55, 0.61, 0.49, 0.054],
    [0.089, -0.04, -0.035, 0.011, 0.034, 0.1, -0.18, 0.12, 0.0012, -0.063]
]).T


def mle_single(tarray):
    mean = np.mean(tarray)
    variance = ((tarray-mean) ** 2).sum()/(len(tarray)-1)
    return mean, variance


def mle_two(tarray):
    mean = np.mean(tarray, axis=0).reshape(1, 2)
    variance = (np.dot((tarray-mean).T, (tarray-mean)))/(len(tarray[:, 0])-1)
    return mean, variance


def mle_three(tarray):
    mean = np.mean(tarray, axis=0).reshape(1, 3)
    variance = (np.dot((tarray-mean).T, (tarray-mean)))/(len(tarray[:, 0])-1)
    return mean, variance


if __name__ == '__main__':
    print("-----------(a)问题------------")
    #one
    m1, v1 = mle_single(x[:, 0])
    m2, v2 = mle_single(x[:, 1])
    m3, v3 = mle_single(x[:, 2])
    mm = np.array([m1, m2, m3])
    v = np.array([v1, v2, v3])
    print("x1 x2 x3各自均值")
    print(mm)
    print("x1 x2 x3各自方差")
    print(v)
    print("-----------(b)问题-------------")
    # two
    m, v = mle_two(x[:, :2])
    print("x1 x2均值")
    print(m)
    print("x1 x2方差")
    print(v)
    m, v = mle_two(x[:, 1:])
    print("x2 x3均值")
    print(m)
    print("x2 x3方差")
    print(v)
    m, v = mle_two(x[:, [0, 2]])
    print("x1 x3均值")
    print(m)
    print("x1 x3方差")
    print(v)

    print("-----------(c)问题------------")
    m, v = mle_three(x)
    print("x1 x2 x3均值")
    print(m)
    print("x1 x2 x3方差")
    print(v)
    print("-----------(d)问题------------")
    m, v = mle_three(w)
    print("w2中x1 x2 x3均值")
    print(m)
    print("w2中x1 x2 x3方差")
    print(np.diag(v))
