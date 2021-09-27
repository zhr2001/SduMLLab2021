import numpy as np
from matplotlib import pyplot as plt

x = np.loadtxt('../data3/ex3Logx.dat', delimiter=',', dtype=float)
y = np.loadtxt('../data3/ex3Logy.dat', dtype=float)
m = len(y)

positiveClass = []
negativeClass = []

List = np.nditer(y, flags=['f_index'])
for i in List:
    if i:
        positiveClass.append(List.index)
    else:
        negativeClass.append(List.index)

fig, ax = plt.subplots()
ax.scatter(x[positiveClass][:, 0], x[positiveClass][:, 1])
ax.scatter(x[negativeClass][:, 0], x[negativeClass][:, 1])
plt.show()


##########################################################
#   Calculate divided coil
##########################################################


def productFeatureVector(u, v):
    a = []
    for Index in [0, 1, 2, 3, 4, 5, 6]:
        for item in np.arange(0, Index + 1, 1):
            a.append(pow(u, item) * pow(v, Index - item))

    return a


def hZeta(xVector, zeta):
    return 1 / (1 + np.exp(np.dot(-zeta.reshape((1, len(zeta))), xVector)[0]))


def Loss(zeta, lam):
    loss = 0
    R = np.nditer(range(m), ['f_index'])
    for item in R:
        loss += y[R.index] * np.log(hZeta(x[R.index, :], zeta)) + (1 - y[R.index]) * (
                    1 - np.log(hZeta(x[R.index, :], zeta)))

    loss = -loss / m
    loss += lam / (2*m) * (np.linalg.norm(zeta) ** 2)
    return loss


def gradient(zeta, lam):
    g = []
    const = []

    for item in range(m):
        const.append(hZeta(x[item, :], zeta) - y[item])

    const = np.array(const) / m

    for Index in range(28):
        if Index == 0:
            g.append(np.dot(const, x[:, Index]))
        else:
            g.append((np.dot(const, x[:, Index]) + lam / m * zeta[Index])[0])

    return np.array(g).reshape(28, 1)


def Newton(zeta, lam):
    init = np.zeros((len(zeta), len(zeta)))
    for Index in range(m):
        s = x[Index, :]
        s = s.reshape(1, 28)
        init += hZeta(x[Index, :], zeta) * (1 - hZeta(x[Index, :], zeta)) * s.transpose() * s
    init /= m

    temp = np.identity(len(zeta))
    temp[0][0] = 0
    init += lam / m * temp

    return init


class Solve:
    __r = 0.0
    __num = 0

    def __init__(self, learningRate):
        self.__r = learningRate

    def getResult(self):
        print("Learning Rate     : ", self.__r)
        zeta = np.zeros((28, 1))
        flag = 1e9
        LL = Loss(zeta, self.__r)
        while abs(flag) > 0.000001:
            self.__num += 1
            zeta -= np.matmul(np.linalg.inv(Newton(zeta, self.__r)), gradient(zeta, self.__r))
            flag = LL - Loss(zeta, self.__r)
            LL -= flag
        print("Iterations num    : ", self.__num)
        print("The minimum value : ", LL)
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        return zeta


transform = []
for index in range(m):
    transform.append(productFeatureVector(x[index, :][0], x[index, :][1]))

x = np.array(transform)

##########################################################
#   Comparison between the three results
##########################################################

u = np.arange(-1, 1.5, 0.0125)
v = np.arange(-1, 1.5, 0.0125)
U = np.nditer(u, flags=['f_index'])
V = np.nditer(v, flags=['f_index'])

z = np.zeros((200, 200))
z1 = np.zeros((200, 200))
z2 = np.zeros((200, 200))

s0 = Solve(0)
Zeta = s0.getResult()

s1 = Solve(1)
Zeta1 = s1.getResult()

s2 = Solve(10)
Zeta2 = s2.getResult()

for itemU in U:
    for itemV in V:
        z[U.index, V.index] = np.dot(productFeatureVector(itemU, itemV), Zeta)

for itemU in U:
    for itemV in V:
        z1[U.index, V.index] = np.dot(productFeatureVector(itemU, itemV), Zeta1)

for itemU in U:
    for itemV in V:
        z2[U.index, V.index] = np.dot(productFeatureVector(itemU, itemV), Zeta2)

# plt(u, v, z)
# plt.show()
