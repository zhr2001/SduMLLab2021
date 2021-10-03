from PIL import Image
import numpy as np

im = np.array(Image.open('../data6/ex8Data/bird_large.tiff'), dtype=float)
nums = im.shape[0] ** 2
width = im.shape[0]


class KMean:
    def __init__(self, k):
        self.__K = k
        self.u = np.random.randint(256, size=(self.__K, 3))
        self.c = np.zeros((nums, 1))
        self.select = [] * self.__K

    def updateU(self, uIndex):
        temp = np.zeros((1, 3), dtype=float)
        cnt = 0
        for color in self.select[uIndex]:
            cnt += 1
            row = int(color / width)
            col = color - row * width
            temp += im[row][col]

        if cnt != 0:
            print(uIndex, " will be modify")
            self.u[uIndex] = temp / cnt

    def findC(self, xIndex):
        minIndex = 0

        row = int(xIndex / width)
        col = xIndex - row * width

        minValue = np.linalg.norm(self.u[0, :] - im[row][col])

        for item in range(16):
            # print(self.u[item, :] - im[row][col])
            temp = np.linalg.norm(self.u[item, :] - im[row][col])
            if temp < minValue:
                minIndex = item
                minValue = temp

        self.c[xIndex] = minIndex
        self.select[minIndex].append(xIndex)

    def Solve(self):
        for iteration in range(70):
            self.select = []
            for item in range(self.__K):
                self.select.append([])

            for indexIm in range(nums):
                self.findC(indexIm)

            for indexU in range(self.__K):
                self.updateU(indexU)

        for item in range(nums):
            row = int(item / width)
            col = item - row * width

            im[row][int(col)] = self.u[int(self.c[item])]

        np.savetxt('result.c.txt', self.c, fmt='%d')
        Ima = Image.fromarray(np.uint8(im))
        Ima.save('result.jpg')


k = KMean(16)
k.Solve()