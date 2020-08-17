import numpy as np
from sklearn.decomposition import PCA as sk_PCA

from PIL import Image
import time
from sklearn.neighbors import KNeighborsClassifier

h = 112
w = 92
data = np.zeros((40, 10, w * h))  # 储存所有图像
for i in range(40):
    for j in range(10):
        im = Image.open("att_faces/" + "s" + str(i + 1) + "/" + str(j + 1) + ".pgm")
        data[i, j, :] = np.array(im).reshape(-1) / 255

dtrain, dtest = data[:, :5, :], data[:, 5:, :]

# 划分训练集和测试集
X_train, X_test = dtrain.reshape(-1, w * h), dtest.reshape(-1, w * h)  # [样本数量, 样本维度]
y_train, y_test = np.linspace(1, 40, 40).repeat(5), np.linspace(1, 40, 40).repeat(5)
knn = KNeighborsClassifier(n_neighbors=1)


def pca():
    temp = 200
    mmean = np.mean(X_train, 0, keepdims=True)
    centerlized = X_train - mmean

    var = np.dot(centerlized.T, centerlized)
    evalue, evector = np.linalg.eig(var)
    evalue, evector = np.real(evalue), np.real(evector)
    evector = evector[:, np.argsort(-evalue)]

    temp = evector[:, :temp]
    # temp /= np.linalg.norm(temp)

    result1 = np.dot(X_train - mmean, temp)
    result2 = np.dot(X_test - np.mean(X_test, 0, keepdims=True), temp)

    return result1, result2


def betterpca(X):
    temp = 200
    mmean = np.mean(X, 0, keepdims=True)
    centerlized = X - mmean

    var = np.dot(centerlized, centerlized.T)
    evalue, evector = np.linalg.eig(var)
    evalue, evector = np.real(evalue), np.real(evector)
    # topk = np.argsort(evalue)[0:100]
    # temp = evector[topk]


    evector = evector[:, np.argsort(-evalue)]
    evalue = -np.sort(-evalue)
    temp = centerlized.T.dot(evector[:, :temp]).dot(np.diag(np.power(evalue[:temp], -0.5)))
    # temp /= np.linalg.norm(temp)
    result1 = np.dot(X - mmean, temp)
    result2 = np.dot(X_test - np.mean(X_test, 0, keepdims=True), temp)
    result3 = np.dot(X_test, temp)

    return result1, result2, result3, temp


def pca_test(isbetter):
    t_start = time.time()
    if isbetter:
        train_data, test_data, _, _ = betterpca(X_train)
    else:
        train_data, test_data = pca()
    t_end = time.time()
    t = t_end - t_start
    print("耗时: %.2f" % t, "s")

    knn.fit(train_data, y_test)
    y_pred_pca = knn.predict(test_data)

    total = 0
    for i in range(y_test.shape[0]):
        if y_pred_pca[i] == y_test[i]:
            total += 1
    acc = total / y_test.shape[0]
    if isbetter:
        print("使用技巧的PCA分类准确率: ", acc)
    else:
        print("PCA分类准确率: ", acc)


def mda(X, Y):
    sample, feature = X.shape
    classes = np.unique(Y)
    class_num = len(np.unique(Y))

    X_class = {}
    for i in range(sample):
        if Y[i] not in X_class:
            X_class[Y[i]] = X[i, :]
        else:
            X_class[Y[i]] = np.row_stack((X_class[Y[i]], X[i, :]))
    X_mean = X.mean(axis=0)
    class_mean = np.zeros((class_num, feature))
    Sw = np.zeros((feature, feature))
    Sb = np.zeros((feature, feature))
    for i in range(class_num):
        class_mean[i, :] = X_class[classes[i]].mean(axis=0)
        Sw += np.dot((X_class[classes[i]] - class_mean[i, :]).T, X_class[classes[i]] - class_mean[i, :])
        Sb += X_class[classes[i]].shape[0] * (class_mean[[i], :] - X_mean).T.dot(class_mean[[i], :] - X_mean)
    e, u = np.linalg.eig(np.linalg.inv(Sw + 0 * np.eye(feature)).dot(Sb))
    e, u = np.real(e), np.real(u)
    u = u[:, np.argsort(-e)]
    # e = -np.sort(-e)
    res = u[:200].T
    return X.dot(res), X_test.dot(res)


def mda_test():
    t_start = time.time()
    train_data, test_data = mda(X_train, y_train)
    t_end = time.time()
    t = t_end - t_start
    print("耗时: %.2f" % t, "s")

    knn.fit(train_data, y_test)
    y_pred_mda = knn.predict(test_data)

    total = 0
    for i in range(y_test.shape[0]):
        if y_pred_mda[i] == y_test[i]:
            total += 1
    acc = total / y_test.shape[0]
    print("MDA分类准确率: ", acc)


def dpdr():
    t_start = time.time()
    train_1, train_2 = np.linalg.qr(X_train.T, mode="reduced")
    t_end = time.time()
    t = t_end - t_start
    print("耗时: %.2f" % t, "s")
    train_data = np.dot(X_train, train_1)
    test_data = np.dot(X_test, train_1)

    knn.fit(train_data, y_test)
    y_pred_dpdr = knn.predict(test_data)

    total = 0
    for i in range(y_test.shape[0]):
        if y_pred_dpdr[i] == y_test[i]:
            total += 1
    acc = total / y_test.shape[0]
    print("DPDR分类准确率: ", acc)


def extend_pca():
    train_data, _, test_data, temp = betterpca(X_train)
    restruct = np.dot(test_data, temp.T)

    # error1 = np.power(np.linalg.norm(X_test - restruct), 2) / 200
    error1 = 100 * np.linalg.norm(X_test - restruct) / (np.linalg.norm(X_test) + np.linalg.norm(restruct))

    print("PCA ETE: %.4f" % error1, "%")
    train_data2, _, test_data2, temp2 = betterpca(np.r_[X_train, X_test])
    restruct2 = np.dot(test_data2, temp2.T)
    # error2 = np.power(np.linalg.norm(X_test - restruct2), 2) / 200
    error2 = 100 * np.linalg.norm(X_test - restruct2) / (np.linalg.norm(X_test) + np.linalg.norm(restruct2))

    print("PCA ETE+: %.4f" % error2, "%")


def extendDPDR():
    train_1, train_2 = np.linalg.qr(X_train.T, mode="reduced")

    restruct = np.dot(np.matmul(X_test, train_1), train_1.T)
    # error1 = np.power(np.linalg.norm(X_test - restruct), 2) / 200
    error1 = 100 * np.linalg.norm(X_test - restruct) / (np.linalg.norm(X_test) + np.linalg.norm(restruct))
    print("DPDR ETE: %.4f" % error1, "%")

    train_11, train_22 = np.linalg.qr(np.r_[X_train, X_test].T, mode="reduced")

    restruct2 = np.dot(np.dot(X_test, train_11), train_11.T)
    # error2 = np.power(np.linalg.norm(X_test - restruct2), 2) / 200
    error2 = 100 * np.linalg.norm(X_test - restruct2) / (np.linalg.norm(X_test) + np.linalg.norm(restruct2))

    print("DPDR ETE+: %.4f" % error2, "%")


if __name__ == '__main__':
    # pca_test(False)  # PCA
    # pca_test(True)  # 使用技巧的PCA
    mda_test()
    # dpdr()
    # extend_pca()
    # extendDPDR()
