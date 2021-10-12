import random
import numpy as np
from PIL import Image

baseUrl = '../data/orl_faces/s'
sample_index = []
test_index = []
dirNum = 40


def produceImage(dirIndex, imgIndex):
    return np.array(Image.open(baseUrl + str(dirIndex) + '/' + str(imgIndex) + '.pgm'))


def PCA(target):



for i in range(dirNum):
    temp = np.random.choice(range(1, 11), random.randint(5, 7), replace=False)
    sample_index.append(temp)

    flag = [False] * 10
    for item in temp:
        flag[item - 1] = True
    temp_test = []
    for item in range(10):
        if not flag[item]:
            temp_test.append(item + 1)

    test_index.append(temp_test)

sample = []
test = []

sampleY = []
realY = []

for item in range(1, dirNum+1):
    indexSample = sample_index[item - 1]
    testSample = test_index[item - 1]

    for index in indexSample:
        sample.append(produceImage(item, index))
        sampleY.append(item)

    for index in testSample:
        test.append(produceImage(item, index))
        realY.append(item)

sampleY = np.array(sampleY)
realY = np.array(realY)
sample = np.array(sample)
test = np.array(test)

for item in sample:
    PCA(item)