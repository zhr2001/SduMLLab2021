import numpy as np
from matplotlib import pyplot as plt

key = [1000, 2000, 2960, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
accuracy = []

for item in key:
    temp = 0

    x = np.loadtxt('testNum'+str(item)+'Result.txt')
    real = x[:, 9]
    predict = x[:, 10]

    for index in range(len(real)):
        if real[index] == predict[index]:
            temp += 1

    accuracy.append(temp/len(real))

accuracy = np.array(accuracy)
print(accuracy)