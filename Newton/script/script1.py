import numpy as np
from matplotlib import pyplot as plt
import math

x = np.loadtxt('../data2/ex2x.dat', dtype=float)
y = np.loadtxt('../data2/ex2y.dat', dtype=float)
m = len(y)
y = y.reshape(m, 1)

positiveClass = []
negativeClass = []

it = np.nditer(y, flags=['f_index'])
for item in it:
    if item == 1:
        positiveClass.append(it.index)
    else:
        negativeClass.append(it.index)

positiveClass = np.array(positiveClass)
negativeClass = np.array(negativeClass)

fig, ax = plt.subplots()
ax.scatter(x[positiveClass][:, 0], x[positiveClass][:, 1])
ax.scatter(x[negativeClass][:, 0], x[negativeClass][:, 1])


######################################################################
#   Calculate the divided line to classify
######################################################################

def hZeta(zeta):
    return 1.0/(1.0 + np.exp(-np.matmul(x, zeta)))


def Loss(zeta):
    hz = hZeta(zeta)
    result = 0.
    IT = np.nditer(hz, flags=['f_index'])

    for ITEM in IT:
        if y[IT.index]:
            result += -1 * math.log(ITEM)

        else:
            result += -1 * math.log(1 - ITEM)

    return result / m


def gradient(zeta, alpha):
    temp = np.nditer(hZeta(zeta) - y, flags=['f_index'])
    res = np.zeros((1, x.shape[1]))

    for Item in temp:
        res += Item * x[temp.index, :]

    return -1*alpha * 1/m * res


def NewtonGradient(zeta):
    hZ = np.nditer(hZeta(zeta), flags=['f_index'])
    H = np.zeros((3, 3))
    for Item in hZ:
        temp = x[hZ.index, :]
        H += Item * (1-Item) * np.matmul(temp.reshape(3, 1), temp.reshape(1, 3))

    return np.matmul(np.linalg.inv(1/m*H), gradient(zeta, -1).reshape(3, 1))


x = np.c_[np.ones((m, 1), dtype=float), x]
Zeta = np.zeros(x.shape[1], dtype=float).reshape(x.shape[1], 1)
flag = 1e9
loss = Loss(Zeta)

print("000 Loss: ", loss)

num = 0

while abs(flag) > 0.000001:
    num += 1
    # Zeta += np.transpose(gradient(Zeta, 0.001))
    Zeta -= NewtonGradient(Zeta)
    # print(y - np.dot(x, Zeta))
    # print(Zeta)

    flag = Loss(Zeta) - loss
    loss = flag + loss
    print("loss: ", Loss(Zeta))
    print("flag: ", flag)

print("The result: ", Zeta)
print("Iteration nums: ", num)

#####################################################################
#   Draw divided line
#####################################################################
a = Zeta[0] / -Zeta[1]
b = Zeta[0] / -Zeta[2]

plt.plot([a, 0], [0, b], 'r')
plt.xlim(10, 70)
plt.ylim(40, 100)
plt.xlabel('exam1 scores')
plt.ylabel('exam2 scores')
plt.show()

# Normal:


# Newton
# loss:  0.4054474249282528
# The result:  [[-16.37873983]
#  [  0.14834074]
#  [  0.15890842]]