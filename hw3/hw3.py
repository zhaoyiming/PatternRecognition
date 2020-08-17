import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

def generatedata(n):
    xdata = np.zeros((100, n))
    ydata = np.zeros((100, n))
    for i in range(100):
        x = 2 * np.random.rand(n) - 1
        # x = np.random.uniform(-1, 1, n)
        noise = np.random.normal(0, np.sqrt(0.1), n)
        y = np.power(x, 2) + noise

        xdata[i] = x
        ydata[i] = y
    return xdata, ydata


def res12(temp, n):
    x, y = generatedata(n)

    bias = np.zeros((100, 1))
    variance = np.zeros((100, 1))
    bias = np.mean((temp - np.power(x, 2)), 1, keepdims=True)
    variance = 0 * variance
    error = np.power(bias, 2) + variance
    mean_bias = np.mean(bias, 0)
    mean_variance = np.mean(variance, 0)
    print('当n=' + str(n) + ', g(x)=' + str(temp) + '时,', end="")
    print(" 偏差均值:", mean_bias[0], end="")
    print(", 方差均值:", mean_variance[0])
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.hist(error)
    ax.title.set_text('n=' + str(n) + ', g(x)=' + str(temp) + ' 误差直方图')
    ax2 = fig.add_subplot(122)
    plt.xlabel('iterators')
    plt.ylabel('bias/variance')
    ax2.plot(bias, "+-", label="bias")
    ax2.plot(variance, label="variance")
    ax2.title.set_text('n=' + str(n) + ', g(x)=' + str(temp) + ' 偏差/方差变化图')

    ax2.legend()

    plt.savefig('data/hw3-g(x)=' + str(temp) + '-n=' + str(n) + '.png', bbox_inches='tight')
    plt.show()


def res3(n):
    x, y = generatedata(n)
    bias = np.zeros((100, 1))
    variance = np.zeros((100, 1))

    a = np.zeros((100, 2))
    gx = np.zeros((100, n))

    for i in range(100):
        a[i, :] = np.polyfit(x[i, :], y[i, :], 1)
        for j in range(n):
            gx[i, j] = a[i, 1] + a[i, 0] * x[i, j]
    bias = np.mean((gx - np.power(x, 2)), axis=1, keepdims=True)
    for i in range(100):
        variance[i] = np.var(gx[i, :])

    error = np.square(bias) + variance
    mean_bias = np.mean(np.power(bias, 2), 0)
    mean_variance = np.mean(variance, 0)

    print('当n=' + str(n) + ', g(x)=a0+a1*x时,', end="")
    print(" 偏差均值:", mean_bias[0], end="")
    print(", 方差均值:", mean_variance[0])
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.hist(error)
    ax.title.set_text('n=' + str(n) + ', g(x)=a0+a1*x 误差直方图')

    ax2 = fig.add_subplot(122)
    plt.xlabel('iterators')
    plt.ylabel('bias/variance')
    ax2.plot(bias, "+-", label="bias")
    ax2.plot(variance, label="variance")
    ax2.title.set_text('n=' + str(n) + ', g(x)=a0+a1*x 偏差/方差变化图')

    ax2.legend()

    plt.savefig('data/hw3-g(x)=a1-n=' + str(n) + '.png', bbox_inches='tight')
    plt.show()


def res4(n):
    x, y = generatedata(n)
    bias = np.zeros((100, 1))
    variance = np.zeros((100, 1))

    a = np.zeros((100, 4))
    gx = np.zeros((100, n))

    for i in range(100):
        a[i, :] = np.polyfit(x[i, :], y[i, :], 3)
        for j in range(n):
            gx[i, j] = a[i, 3] + a[i, 2] * x[i, j] + a[i, 1] * (x[i, j] ** 2) + a[i, 0] * (x[i, j] ** 3)
    bias = np.mean((gx - np.power(x, 2)), 1, keepdims=True)
    for i in range(100):
        variance[i] = np.var(gx[i, :])

    error = np.power(bias, 2) + variance
    mean_bias = np.mean(bias, 0)
    mean_variance = np.mean(variance, 0)
    print('当n=' + str(n) + ', g(x)=a0+a1*x+a2*x^2+a3*x^3时,', end="")
    print(" 偏差均值:", mean_bias[0], end="")
    print(", 方差均值:", mean_variance[0])
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.hist(error)
    ax.title.set_text('n=' + str(n) + ', g(x)=a0+a1*x+a2*x^2+a3*x^3 误差直方图')
    ax2 = fig.add_subplot(122)
    plt.xlabel('iterators')
    plt.ylabel('bias/variance')
    ax2.plot(bias, "+-", label="bias")
    ax2.plot(variance, label="variance")
    ax2.title.set_text('偏差/方差变化图')
    ax2.legend()

    plt.savefig('data/hw3-g(x)=a3=' + str(n) + '.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    res12(0.5, 10)
    res12(1, 10)
    res3(10)
    res4(10)
    res12(0.5, 100)
    res12(1, 100)
    res3(100)
    res4(100)
