import numpy as np
from matplotlib import pyplot as plt

x = np.loadtxt('../data3/ex3Linx.dat', dtype=float)
y = np.loadtxt('../data3/ex3Liny.dat', dtype=float)
m = len(x)

featureMatrix = np.identity(6)
featureMatrix[0][0] = 0


def produceZeta(lam):
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(x), x) + lam * featureMatrix), np.transpose(x)), y.reshape(m, 1))


def prediction(X, theta):
    return np.matmul(X, theta)


x = np.c_[np.ones((m, 1)), x.reshape(m, 1)]
t = x[:, 1].reshape(m, 1)
for i in np.arange(2, 6, 1):
    x = np.c_[x, np.power(t, i)]

zeta0 = produceZeta(0)
zeta1 = produceZeta(1)
zeta10 = produceZeta(10)

fig, ax = plt.subplots()
ax.scatter(x[:, 1], y)

sample = np.arange(-1, 1, 0.01).reshape(200, 1)
sample = np.c_[np.ones((200, 1)), sample]
t = sample[:, 1]
for i in np.arange(2, 6, 1):
    sample = np.c_[sample, np.power(t, i)]

plt.plot(t, prediction(sample, zeta0))
plt.plot(t, prediction(sample, zeta1))
plt.plot(t, prediction(sample, zeta10))
plt.show()

