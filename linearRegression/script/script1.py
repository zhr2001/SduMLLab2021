import numpy as np
from matplotlib import pyplot as plt

alpha = 0.05
x = np.loadtxt('../data1/ex1_1x.dat')
y = np.loadtxt('../data1/ex1_1y.dat')
m = len(x)
plt.scatter(x, y)

x = np.array([np.ones(m), x])


def HZeta(zeta, X):
    return np.dot(zeta, X)


def getDown(zeta):
    res = []

    for j in np.arange(0, len(zeta), 1):
        temp = 0
        for i in np.arange(0, m, 1):
            temp -= alpha * (1 / m) * (HZeta(zeta, x[:, i]) - y[i]) * x[:, i][j]
        res.append(temp)

    return res


def judgeStop(down):
    return np.linalg.norm(down) < 0.000001


Zeta = np.array([0, 0])
Down = getDown(Zeta)

while judgeStop(Down) != 1:
    Zeta = Zeta + Down
    Down = getDown(Zeta)

result = []

for i in np.arange(0, m, 1):
    result.append(HZeta(Zeta, x[:, i]))

x = np.array(np.ravel(x[1:, ]))
plt.plot(x, result, 'r', x, y, 'y')
plt.show()

print("x = 3.5 prediction: ", HZeta(Zeta, [1, 3.5]))
print("x = 7   prediction: ", HZeta(Zeta, [1, 7]))
