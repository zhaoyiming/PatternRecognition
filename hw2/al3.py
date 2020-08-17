import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

max_iteration = 1000


def batchperceptron(train_data, test_data):
    w = np.array([0, 0, 0, 0, 0.1]).T
    error_index = np.where(np.dot(train_data, w) < 0)[0]
    error_data = train_data[error_index]
    err = np.sum(error_data, axis=0)
    J = np.dot(err, err)

    index = 1
    while True:
        index += 1
        delta = (1 / index) * err
        w = w + delta.T
        error_data = train_data[np.where(np.dot(train_data, w) < 0)[0]]
        err = np.sum(error_data, axis=0)
        J = np.sqrt(np.dot(err, err))
        if index > max_iteration:
            break
        if J < 0.001:
            break
    acc = 1 - len(np.where(np.dot(test_data, w) < 0)[0]) / 50
    return acc


def fixed_perceptron(train_data, test_data):
    w = np.array([0, 0, 0, 0, 0.1]).T
    error_index = np.where(np.dot(train_data, w) < 0)[0]
    error_data = train_data[error_index]
    J = error_data.shape[0]
    index = 1
    while True:
        index += 1
        for i in range(J):
            w = w + error_data[i, :].T
        error_data = train_data[np.where(np.dot(train_data, w) < 0)[0]]
        J = error_data.shape[0]

        if index > max_iteration:
            break
        if J <= 0:
            break
    acc = 1 - len(np.where(np.dot(test_data, w) < 0)[0]) / 50
    return acc


def margin_perceptron(train_data, test_data):
    w = np.array([0, 0, 0, 0, 0.1]).T
    m = 0.01
    error_index = np.where(np.dot(train_data, w) < -m)[0]
    error_data = train_data[error_index]
    J = error_data.shape[0]

    index = 1
    while True:
        index += 1
        for i in range(J):
            w = w + error_data[i, :].T
        error_data = train_data[np.where(np.dot(train_data, w) < -m)[0]]
        J = error_data.shape[0]
        if index > max_iteration:
            break
        if J <= 0:
            break
    acc = 1 - len(np.where(np.dot(test_data, w) < 0)[0]) / 50
    return acc


def batchincre_perceptron(train_data, test_data):
    w = np.array([0, 0, 0, 0, 0.1]).T
    error_index = np.where(np.dot(train_data, w) < 0)[0]
    error_data = train_data[error_index]
    err = np.sum(error_data, axis=0)
    J = np.dot(err, err)

    index = 1
    while True:
        index += 1
        delta = (1 / index) * err
        w = w + delta.T
        error_data = train_data[np.where(np.dot(train_data, w) < 0)[0]]
        err = np.sum(error_data, axis=0)
        J = np.sqrt(np.dot(delta, delta))
        if index > max_iteration:
            break
        if J < 0.001:
            break

    acc = 1 - len(np.where(np.dot(test_data, w) < 0)[0]) / 50
    return acc


def balance_window(traindata, testdata):
    traindata[np.where(traindata < 0)[0]] = -traindata[np.where(traindata < 0)[0]]
    testdata[np.where(testdata < 0)[0]] = -testdata[np.where(testdata < 0)[0]]

    w = np.array([1, 1, 1, 1, 1]).T / 2
    ww = -w
    alpha = 3
    length = 1
    index = 1
    while True:
        index += 1
        for i in range(50):
            d = np.dot(traindata[i, :], w) - traindata[i, :] * ww
            if d.any() < 0:
                if i < 25:
                    for j in range(5):
                        y = traindata[i, j]
                        w[j] = np.dot(np.power(alpha, y), w[j])
                        ww[j] = np.dot(np.power(alpha, -y), ww[j])
                if i >= 25:
                    for j in range(5):
                        y = traindata[i, j]
                        ww[j] = np.dot(np.power(alpha, y), ww[j])
                        w[j] = np.dot(np.power(alpha, -y), w[j])

        error_data = traindata[np.where((np.dot(traindata, w) - np.dot(traindata, ww)) < 0)[0]]
        length = error_data.shape[0]
        if length <= 0:
            break
        if index > max_iteration:
            break
    temp = np.dot(traindata, w) - np.dot(traindata, ww)
    acc = 1 - len(np.where(temp < 0)[0]) / 50

    return acc


def single_relaxation(train_data, test_data):
    w = np.array([0, 0, 0, 0, 0.1]).T
    m = 0.01
    error_index = np.where(np.dot(train_data, w) < -m)[0]
    error_data = train_data[error_index]
    J = error_data.shape[0]

    index = 1
    while True:
        index += 1
        for i in range(J):
            y = error_data[i, :]
            w = w + (1 / index) * (m - np.dot(y, w)) * y.T / np.dot(y, y)
        error_data = train_data[np.where(np.dot(train_data, w) < -m)[0]]
        J = error_data.shape[0]
        if index > max_iteration:
            break
        if J < 0:
            break

    acc = 1 - len(np.where(np.dot(test_data, w) < 0)[0]) / 50
    return acc


def lms(train_data, test_data):
    w = np.array([0, 0, 0, 0, 0.1])
    m = np.ones((1, 50))
    k = 0
    J = [1, 1, 1, 1, 1]

    index = 1
    while True:
        if np.linalg.norm(J) < 0.001:
            break
        index += 1

        k = np.mod(k + 1, 49)
        if k == 0:
            k = 49
        J = (1 / index) * (m[:, k] - np.matmul(w, train_data[k, :].T)) * train_data[k, :]
        w = w + J
        w = w / np.linalg.norm(w)

        if index > max_iteration:
            break

    acc = 1 - len(np.where(np.dot(test_data, w.T) < 0)[0]) / 50
    return acc


def hk(train_data, test_data):
    w = np.array([[0, 0, 0, 0, 0.1]])
    m = np.ones((1, 50))
    e = m
    index = 1
    while True:
        if len(np.where(np.abs(e) > 0.001)[0]) < 0:
            break
        index += 1
        e = np.matmul(train_data, w.T).T - m
        ee = (e + np.abs(e)) / 2
        m = e + (2 / index) * ee
        w = np.matmul(m, np.linalg.pinv(train_data).T)

        if index > max_iteration:
            break

    acc = 1 - len(np.where(np.dot(test_data, w.T) < 0)[0]) / 50
    return acc


def betterhk(train_data, test_data):
    w = np.array([[0, 0, 0, 0, 0.1]])
    m = np.ones((1, 50))
    e = m
    index = 1
    while True:
        if len(np.where(np.abs(e) > 0.001)[0]) < 0:
            break
        index += 1
        e = np.matmul(train_data, w.T).T - m
        ee = (e + np.abs(e)) / 2
        m = m + (1 / index) * (e + np.abs(e))
        w = np.matmul(m, np.linalg.pinv(train_data).T)
        if index > max_iteration:
            break
    acc = 1 - len(np.where(np.dot(test_data, w.T) < 0)[0]) / 50
    return acc


def res(newd1, newd2):
    res = np.zeros((100, 9))
    for i in range(100):
        train_data_1, test_data_1 = train_test_split(newd1, test_size=0.5)
        train_data_2, test_data_2 = train_test_split(newd2, test_size=0.5)
        train_data, test_data = np.concatenate((train_data_1, train_data_2), axis=0), np.concatenate(
            (test_data_1, test_data_2), axis=0)
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)
        res[i, 0] = batchperceptron(train_data, test_data)
        res[i, 1] = fixed_perceptron(train_data, test_data)
        res[i, 2] = margin_perceptron(train_data, test_data)
        res[i, 3] = batchincre_perceptron(train_data, test_data)
        res[i, 5] = single_relaxation(train_data, test_data)
        res[i, 6] = lms(train_data, test_data)
        res[i, 7] = hk(train_data, test_data)
        res[i, 8] = betterhk(train_data, test_data)
        res[i, 4] = balance_window(train_data, test_data)  # chudawenti
    mean_data = np.mean(res, axis=0)
    var_data = np.var(res, axis=0)
    print("批处理感知器算法精度均值：", mean_data[0], "精度方差", var_data[0])
    print("固定增量单样本感知器精度均值：", mean_data[1], "精度方差", var_data[1])
    print("带裕量的变增量感知器精度均值：", mean_data[2], "精度方差", var_data[2])
    print("批处理变增量感知器精度均值：", mean_data[3], "精度方差", var_data[3])
    print("平衡window算法精度均值：", mean_data[4], "精度方差", var_data[4])
    print("单样本裕量松弛算法精度均值：", mean_data[5], "精度方差", var_data[5])
    print("LMS算法精度均值：", mean_data[6], "精度方差", var_data[6])
    print("Ho-Kashyap精度均值：", mean_data[7], "精度方差", var_data[7])
    print("修改Ho-Kashyap精度均值：", mean_data[8], "精度方差", var_data[8])



if __name__ == '__main__':
    data = pd.read_csv("iris.data", header=None)
    data1 = np.array(data.iloc[0:50, 0:4])
    data2 = np.array(data.iloc[50:100, 0:4])
    data3 = np.array(data.iloc[100:150, 0:4])
    arr = np.ones((50, 1))
    newd1 = np.concatenate((data1, arr), axis=1)
    newd2 = np.concatenate((data2, arr), axis=1)
    newd3 = -np.concatenate((data3, arr), axis=1)

    print("（a）: class1 和 class3")
    res(newd1, newd3)
    print("（b）: class2 和 class3")
    res(newd2, newd3)
    # res = np.zeros((100, 9))
    # for i in range(100):
    #     train_data_1, test_data_1 = train_test_split(newd1, test_size=0.5)
    #     train_data_2, test_data_2 = train_test_split(newd3, test_size=0.5)
    #     train_data, test_data = np.concatenate((train_data_1, train_data_2), axis=0), np.concatenate(
    #         (test_data_1, test_data_2), axis=0)
    #     np.random.shuffle(train_data)
    #     np.random.shuffle(test_data)
    #     res[i, 0] = batchperceptron(train_data, test_data)
    #     res[i, 1] = fixed_perceptron(train_data, test_data)
    #     res[i, 2] = margin_perceptron(train_data, test_data)
    #     res[i, 3] = batchincre_perceptron(train_data, test_data)
    #     res[i, 5] = single_relaxation(train_data, test_data)
    #     res[i, 6] = lms(train_data, test_data)
    #     res[i, 7] = hk(train_data, test_data)
    #     res[i, 8] = betterhk(train_data, test_data)
    #     res[i, 4] = balance_window(train_data, test_data) #chudawenti
    # mean_data = np.mean(res, axis=0)
    # var_data = np.var(res, axis=0)
    # print("批处理感知器算法精度均值：", mean_data, "精度方差", var_data)
    #
    # res2 = np.zeros((100, 9))
    # for i in range(100):
    #     train_data_1, test_data_1 = train_test_split(newd2, test_size=0.5)
    #     train_data_2, test_data_2 = train_test_split(newd3, test_size=0.5)
    #     train_data, test_data = np.concatenate((train_data_1, train_data_2), axis=0), np.concatenate(
    #         (test_data_1, test_data_2), axis=0)
    #     np.random.shuffle(train_data)
    #     np.random.shuffle(test_data)
    #     res2[i, 0] = batchperceptron(train_data, test_data)
    #     res2[i, 1] = fixed_perceptron(train_data, test_data)
    #     res2[i, 2] = margin_perceptron(train_data, test_data)
    #     res2[i, 3] = batchincre_perceptron(train_data, test_data)
    #     res2[i, 5] = single_relaxation(train_data, test_data)
    #     res2[i, 6] = lms(train_data, test_data)
    #     res2[i, 7] = hk(train_data, test_data)
    #     res2[i, 8] = betterhk(train_data, test_data)
    #     res2[i, 4] = balance_window(train_data, test_data)
    # mean_data2 = np.mean(res2, axis=0)
    # var_data2 = np.var(res2, axis=0)
    # print("批处理感知器2算法精度均值：", mean_data2, "精度方差", var_data2)
