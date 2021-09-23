import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

x = np.loadtxt('../data1/ex1_1x.dat')
y = np.loadtxt('../data1/ex1_1y.dat')
m = len(x)


def HZeta(zeta, val):
    return np.dot(zeta, np.array([1, val]))


def Loss(zeta):
    res = 0
    for index in np.arange(0, m, 1):
        res += 1 / (2*m) * (HZeta(zeta, x[index]) - y[index]) ** 2
    return res


J_val = np.zeros((100, 100), float)
theta0_val = np.arange(-3, 3, 0.06)
theta1_val = np.arange(-1, 1, 0.02)

index1 = 0
index2 = 0
for itemZ in theta0_val:
    for itemO in theta1_val:
        Zeta = [itemZ, itemO]
        J_val[index1, index2] = Loss(Zeta)
        index2 += 1
    index1 += 1
    index2 = 0

fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.plot_surface(theta1_val, theta0_val, J_val, rstride=1, cstride=1, cmap=cm.viridis)

plt.show()