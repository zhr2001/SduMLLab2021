import numpy as np
from matplotlib import pyplot as plt
import qpsolvers as qp

def getInformation(path, number = 1000, sample = 1):
    Y = []
    cnt = 0
    information = []
    all = 12665

    if sample != 1:
        all = 2115
        number = 2115

    sample_index = np.random.choice(range(all), number, replace=False)

    f = open(path)
    s = 0
    for row in f:
        if s in sample_index:
            information.append([])
            tmp = row.split(' ')
            if tmp[0] == '-1':
                Y.append(-1)
            else:
                Y.append(1)

            for index in range(1, len(tmp)-1):
                t = tmp[index].split(':')
                information[cnt].append([int(t[0]), int(t[1])])

            information[cnt] = np.array(information[cnt])
            information[cnt] = LBP(information[cnt])

            cnt  += 1
        s += 1
    return np.array(information)/10, np.array(Y)


def LBP(info):
    s = [0] * 256
    m = np.zeros((28, 28))
    for item in info:
        row = int(item[0] / 28)
        col = int(item[0] - 28 * row)
        m[row, col] = item[1]

    res = np.zeros((28, 28))

    for i in range(28):
        for j in range(28):
            s[getValue(m, i, j)] += 1
            res[i, j] = getValue(m, i, j)

    return s


def getValue(m, i, j):
    res = 0
    if i==0 or j==0 or i==27 or j==27:
        return 0

    if m[i-1][j-1] < m[i][j]:
        res += 0
    else:
        res += pow(2, 0)

    if m[i-1][j] < m[i][j]:
        res += 0
    else:
        res += pow(2, 1)

    if m[i-1][j+1] < m[i][j]:
        res += 0
    else:
        res += pow(2, 2)

    if m[i][j+1] < m[i][j]:
        res += 0
    else:
        res += pow(2, 3)

    if m[i+1][j+1] < m[i][j]:
        res += 0
    else:
        res += pow(2, 4)

    if m[i+1][j] < m[i][j]:
        res += 0
    else:
        res += pow(2, 5)

    if m[i+1][j-1] < m[i][j]:
        res += 0
    else:
        res += pow(2, 6)

    if m[i][j-1] < m[i][j]:
        res += 0
    else:
        res += pow(2, 7)

    return int(res)

def formulate(x, y):
    return np.dot(x, y)


class Solve:
    def __init__(self, trainIndex=1, testIndex=1, C=0.1, affix = 10, testImage = 0, cal=formulate):
        self.__C = C
        self.__Affix = affix
        self.__cal = cal
        if testImage == 0:
            self.__train_data = np.loadtxt('../data5/data5/training_'+str(trainIndex)+'.txt', dtype=float)
            self.__test_data = np.loadtxt('../data5/data5/test_'+str(testIndex)+'.txt', dtype=float)
            self.__train_Y = self.__train_data[:, 2]
            self.__test_Y = self.__test_data[:, 2]
            self.__train_data = self.__train_data[:, [0, 1]]
            self.__test_data = self.__test_data[:, [0, 1]]

        else:
            self.__train_data, self.__train_Y = getInformation('../data5/data5/train-01-images.svm')
            self.__test_data, self.__test_Y = getInformation('../data5/data5/test-01-images.svm', sample=0)

        print("trainSet: training_" + str(trainIndex) + '.txt')
        print("testSet: test_" + str(testIndex) + '.txt')

        self.__train = len(self.__train_Y)
        self.__test = len(self.__test_Y)

        self.__dividedVector = np.zeros((1, self.__train_data.shape[1]))

        self.__b = 0

    def solution(self):
        x = self.__train_data
        y = self.__train_Y
        q = -np.ones((self.__train, ))
        p = np.zeros((self.__train, self.__train))
        for i in range(self.__train):
            for j in range(0, i+1):
                tmp = self.__Affix
                if i != j:
                    tmp = 0
                p[i][j] = y[i]*y[j]*self.__cal(x[i, :], x[j, :]) + tmp
                p[j][i] = p[i][j]

        self.__matrix = p;

        return qp.solve_qp(P=p, q=q, A=y, b=np.array([0]), lb=np.zeros((self.__train, )),
                           ub=np.full((self.__train, ), self.__C))

    def getWAndB(self):
        C = self.solution()

        cnt = 0
        for index in range(self.__train):
            self.__dividedVector += C[index] * self.__train_Y[index] * self.__train_data[index, :]

        for index in range(self.__train):
            if C[index] > 0:
                cnt += 1
                self.__b += self.__train_Y[index] - np.dot(self.__dividedVector, self.__train_data[index, :])[0]

        if cnt != 0: self.__b /= cnt

        print("get w: ", self.__dividedVector)
        print("get b: ", self.__b)

    def draw(self):
        positiveClass = []
        negativeClass = []
        fig, ax = plt.subplots()
        for index in range(self.__train):
            if self.__train_Y[index] == 1:
                positiveClass.append(self.__train_data[index, :])
            else:
                negativeClass.append(self.__train_data[index, :])

        positiveClass = np.array(positiveClass)
        negativeClass = np.array(negativeClass)

        ax.scatter(positiveClass[:, 0], positiveClass[:, 1])
        ax.scatter(negativeClass[:, 0], negativeClass[:, 1])

        x1 = [0, (1-self.__b) / self.__dividedVector[0, 0]]
        y1 = [(1-self.__b) / self.__dividedVector[0, 1], 0]

        x2 = [0, (-1 - self.__b) / self.__dividedVector[0, 0]]
        y2 = [(-1 - self.__b) / self.__dividedVector[0, 1], 0]

        plt.axline((x1[0], y1[0]), (x1[1], y1[1]))
        plt.axline((x2[0], y2[0]), (x2[1], y2[1]))

        plt.show()

    def drawBoundary(self):
        C = self.solution()
        res = []

        for index in range(self.__train):
            temp = 0
            for i in range(self.__train):
                if C[i] > 0:
                    temp += C[i]*self.__train_Y[i]*self.__matrix[index][i]

            self.__b += self.__train_Y[index] - temp

        self.__b /= self.__train

        fig, ax = plt.subplots()
        s = np.max(self.__train_data[:,0]) - np.min(self.__train_data[:,0])
        x = np.arange(np.min(self.__train_data[:,0]), np.max(self.__train_data[:,0]), s/100)
        s = np.max(self.__train_data[:,1]) - np.min(self.__train_data[:,1])
        y = np.arange(np.min(self.__train_data[:,1]), np.max(self.__train_data[:,1]), s/100)

        for itemX in x:
            for itemY in y:
                temp = 0
                for i in range(len(C)):
                    if C[i] > 0:
                        temp += C[i] * self.__train_Y[i] * self.__cal(np.array([itemX, itemY]), self.__train_data[i, :])

                temp += self.__b
                if abs(temp) < 0.01:
                    res.append([itemX, itemY])

        res = np.array(res)
        print(res)
        ax.scatter(res[:, 0], res[:, 1])

        positiveClass = []
        negativeClass = []
        for index in range(self.__train):
            if self.__train_Y[index] == 1:
                positiveClass.append(self.__train_data[index, :])
            else:
                negativeClass.append(self.__train_data[index, :])
        positiveClass = np.array(positiveClass)
        negativeClass = np.array(negativeClass)

        ax.scatter(positiveClass[:, 0], positiveClass[:, 1])
        ax.scatter(negativeClass[:, 0], negativeClass[:, 1])

        plt.show()

    def getAcc(self):
        cnt = 0.
        for index in range(self.__test):
            if self.__test_Y[index] * (np.dot(self.__dividedVector, self.__test_data[index, :])[0]+self.__b) > 0:
                cnt += 1

        print("ACC  : ", cnt / self.__test)
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    def main(self):
        # self.getWAndB()
        # self.draw()
        # self.getAcc()
        self.drawBoundary()


def kernel(x, y):
    return np.exp(-100 * np.linalg.norm(x-y)**2)

s = Solve(trainIndex=3, affix=0, testImage=0, cal=kernel)
s.main()



