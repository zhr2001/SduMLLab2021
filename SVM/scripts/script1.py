import numpy as np
import qpsolvers as qp


class Solve:
    def __init__(self, trainIndex, testIndex, C):
        self.__C = C
        self.__train_data = np.loadtxt('../data5/data5/training_'+str(trainIndex)+'.txt', dtype=float)
        self.__test_data = np.loadtxt('../data5/data5/test_'+str(testIndex)+'.txt', dtype=float)
        self.__train_Y = self.__train_data[:, 2]
        self.__test_Y = self.__test_data[:, 2]
        self.__train_data = self.__train_data[:, [0, 1]]
        self.__test_data = self.__test_data[:, [0, 1]]

        self.__train = len(self.__train_Y)
        self.__test = len(self.__test_Y)

    def solution(self):
        x = self.__train_data
        y = self.__train_Y
        q = -np.ones((self.__train, 1))
        p = np.zeros((self.__train, self.__train))
        for i in range(self.__train):
            for j in range(0, i+1):
                p[i][j] = y[i]*y[j]*np.dot(x[i, :], x[j, :])
                p[j][i] = p[i][j]

        print(p)

        return qp.solve_qp(P=p, q=q, A=y, b=np.array([0]), ub=np.zeros((self.__train, )),
                           lb=np.full((self.__train, ), self.__C))


s = Solve(1, 2, -100000000000000000)

x = s.solution()
print("QP solution: x = {}".format(x))