import numpy as np
import random
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import StratifiedShuffleSplit

dtest = np.random.rand(20, 3)
dtest[:10, 2], dtest[10:, 2] = 1, 2  # 第三列是标记
dlabeled = np.random.rand(100, 3)
dlabeled[:50, 2], dlabeled[50:, 2] = 1, 2


def dis(x, y):
    res = np.sqrt(np.sum(np.power((x - y), 2), axis=1))
    return res


def getneighbor(train, test, k):
    ndis = dis(train, test)
    near = np.argsort(ndis)[:k]
    return near


def getv(near, train_y):
    top_l = [train_y[i] for i in near]
    d = {}
    for i in top_l:
        d[i] = d.get(i, 0) + 1
    d_list = list(d.items())
    d_list.sort(key=lambda x: x[1], reverse=True)
    label = d_list[0][0]
    return label


def predict(train_x, train_y, test_x, k):
    predicted = []
    for i in range(test_x.shape[0]):
        mat = getneighbor(train_x, test_x[i], k)

        label = getv(mat, train_y)
        predicted.append(label)
    return predicted


def train(isRev):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
    for train_index, test_index in split.split(dlabeled, dlabeled[:, -1]):
        train = dlabeled[train_index, :]
        val = dlabeled[test_index, :]
    train_X, train_Y = train[:, 0:2], train[:, 2]
    val_X, val_Y = val[:, 0:2], val[:, 2]
    test_X, test_Y = dtest[:, 0:2], dtest[:, 2]
    error_mat = {}
    best_k = 1
    k = 1
    while True:

        pred_val = predict(train_X, train_Y, val_X, k)
        all_val = 0
        for i in range(len(val_X)):
            if test_Y[i] == pred_val[i]:
                all_val += 1
        error_rate = 1 - (all_val / len(test_X))
        # print("k为:"+ str(k) +" val error rate:", error_rate)
        error_mat[k] = error_rate

        if k >= 5:
            if isRev:
                if error_mat[k - 2] < error_rate and error_mat[k - 2] < error_mat[k - 4]:
                    break
            else:
                if error_mat[k - 2] > error_rate and error_mat[k - 2] > error_mat[k - 4]:
                    break

        k += 2
        if k > 50:
            break
    error_mat = list(error_mat.items())
    best_k = error_mat[-2][0]
    # print(error_mat)

    pred_test = predict(train_X, train_Y, test_X, best_k)
    all_test = 0
    for i in range(len(test_X)):
        if test_Y[i] == pred_test[i]:
            all_test += 1
    error_rate = 1 - (all_test / len(test_X))
    print("k为:" + str(best_k) + " test error rate:", error_rate)


if __name__ == '__main__':
    for i in range(5):
        print("-------------第" + str(i+1) + "次-------------")
        print("通过验证误差的第一个极小值确定的k：")
        train(True)
        print("通过验证误差的第一个极大值确定的k：")
        train(False)
