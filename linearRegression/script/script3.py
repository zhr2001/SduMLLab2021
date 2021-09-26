import numpy as np
from matplotlib import pyplot as plt

x = np.loadtxt('../data1/ex1_2x.dat')
y = np.loadtxt('../data1/ex1_2y.dat')
m = len(y)
numIteration = 100
result = [[], [], []]

x = np.c_[np.ones(m), x]
y = y.reshape(m, 1)


def Loss(zeta):
    temp = np.dot(x, zeta) - y
    res = 1/(2*m)*np.dot(np.transpose(temp), temp)[0]
    return res


def gradient(zeta, alpha):
    return -1*alpha/m*np.dot(np.transpose(np.dot(x, zeta) - y), x)


sigma = x.std(axis=0)
mu = x.mean(axis=0)

x[:, 1] = (x[:, 1] - np.full((1, m), mu[1], float)) / sigma[1]
x[:, 2] = (x[:, 2] - np.full((1, m), mu[2], float)) / sigma[2]


################################################################################
# draw learn rate 0.0005 0.5 0.1   Loss-Iterations graphic
################################################################################

Theta = np.zeros((3, 1))

for iterator in np.arange(1, numIteration+1, 1):
    result[0].append(Loss(Theta))
    Theta += gradient(Theta, 1.5).reshape(3, 1)

Theta = np.zeros((3, 1))

for iterator in np.arange(1, numIteration+1, 1):
    result[1].append(Loss(Theta))
    Theta += gradient(Theta, 0.5).reshape(3, 1)

Theta = np.zeros((3, 1))

for iterator in np.arange(1, numIteration+1, 1):
    result[2].append(Loss(Theta))
    Theta += gradient(Theta, 0.1).reshape(3, 1)

plt.plot(np.arange(1, numIteration+1, 1), result[0], 'r--', result[1], 'y--', result[2], 'b--')
plt.show()

#########################################################################
# Find the rapidest learning rate in [0.1 , 1.28]
#########################################################################

R = np.arange(0.1, 1.28, 0.001)


def findBestRate():
    minimum = []
    nums = []
    for alpha in R:
        num = 0
        Theta = np.zeros((3, 1))
        Theta += gradient(Theta, alpha).reshape(3, 1)
        temp = Loss(Theta)
        flag = 1e9
        while abs(flag) > 0.00000000000001:
            Theta += gradient(Theta, alpha).reshape(3, 1)
            flag = Loss(Theta) - temp
            temp = flag + temp
            num += 1

        nums.append(num)
        minimum.append(temp)

    return minimum, nums


minValue, nums = findBestRate()

plt.plot(R, nums, 'r--')
plt.show()

plt.plot(R, minValue, 'r--')
plt.show()

minIterationNumIndex = 0
minIteration = 1e9

index = 0

while index < len(nums):
    if nums[index] < minIteration:
        minIteration = nums[index]
        minIterationNumIndex = index
    index += 1

print("Find the rapidest learning rate: ", R[minIterationNumIndex])
print("The Iterations: ", minIteration)

##############################################################################
# Calculate result basic on the calculated learning rate
##############################################################################

alpha = R[minIterationNumIndex]
Theta = np.zeros((3, 1))
Theta += gradient(Theta, alpha).reshape(3, 1)
temp = Loss(Theta)
flag = 1e9

while abs(flag) > 0.00000000000001:
    Theta += gradient(Theta, alpha).reshape(3, 1)
    flag = Loss(Theta) - temp
    temp = flag + temp

x1 = (1650-mu[1])/sigma[1]
x2 = (3-mu[2])/sigma[2]
para = np.array([1, x1, x2])

print("1650 square feet and 3 bedroom prediction: ", np.dot(para, Theta)[0])

