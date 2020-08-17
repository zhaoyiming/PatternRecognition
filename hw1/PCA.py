import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)
x1 = np.array([0.42, -0.2, 1.3, 0.39, -1.6, -0.029, -0.23, 0.27, -1.9, 0.87])
x2 = np.array([-0.087, -3.3, -0.32, 0.71, -5.3, 0.89, 1.9, -0.3, 0.76, -1.0])
x = np.stack((x1, x2), axis=0).T

mean = np.mean(x, axis=0, keepdims=True)
variance = (np.dot((x-mean).T, (x-mean)))/(len(x[:, 0])-1)

print(variance)
evalue, evector=np.linalg.eig(variance)
print(evalue)
print(evector)
k=1
topk = np.argsort(evalue)[0:k]
print("topk:")
print(topk)

result = np.matmul(x, evector[topk].T)
print(evector[topk])
print(result)

axe1 = plt.subplot(211)
s1 = axe1.scatter(x1, x2, color='r', s=25, marker="o")
plt.legend([s1], ['A'])

axe2 = plt.subplot(212)
s2 = axe2.scatter(result[:, 0], result[:, 0].shape[0]*[1], color='g', s=25, marker="o")
plt.legend([s2], ['B'])

plt.show()