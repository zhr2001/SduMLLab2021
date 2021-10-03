import numpy as np

x = np.loadtxt('../data4/data4/nursery.data.txt', delimiter=',', dtype=str)
data_num = 12960
att_num = 9


def hZeta(zeta, xVector):
    return np.matmul(xVector, zeta)


def handle_data_set(ds):
    data = []
    for index in range(data_num):
        temp = ds[index, :]
        feature = []

        check = [
            {
                'usual': 0,
                'pretentious': 1,
                'great_pret': 2
            }, {
                'proper': 0,
                'less_proper': 1,
                'improper': 2,
                'critical': 3,
                'very_crit': 4
            }, {
                'complete': 0,
                'completed': 1,
                'incomplete': 2,
                'foster': 3
            }, {
                '1': 0,
                '2': 1,
                '3': 2,
                'more': 3
            }, {
                'convenient': 0,
                'less_conv': 1,
                'critical': 2
            }, {
                'convenient': 0,
                'inconv': 1
            }, {
                'nonprob': 0,
                'slightly_prob': 1,
                'problematic': 2
            }, {
                'recommended': 0,
                'priority': 1,
                'not_recom': 2
            }, {
                'not_recom': 0,
                'recommend': 1,
                'very_recom': 2,
                'priority': 3,
                'spec_prior': 4
            }
        ]

        for Index in range(att_num):
            feature.append(check[Index][temp[Index]])

        data.append(feature)

    data = np.array(data)
    return data


DS = handle_data_set(x)
Y = DS[:, 8]
DS = DS[:, [0, 1, 2, 3, 4, 5, 6, 7]]
DS = np.c_[np.ones((data_num, 1)), DS]


class Solve:

    training_data = []
    test_data = []

    t = []

    feature = []

    def __init__(self, testNum):
        self.__test = testNum
        test_index = np.random.choice(range(data_num), self.__test, replace=False)
        training_index = []

        flag = np.zeros((data_num, 1))
        for item in test_index:
            flag[item] = 1

        s = np.nditer(flag, ['f_index'])
        for item in s:
            if item == 0:
                training_index.append(s.index)

        self.test_data = DS[test_index]
        self.training_data = DS[training_index]
        self.Y = Y[training_index]
        self.y = np.zeros_like(self.Y)
        self.testY = Y[test_index]

    def preSolve(self, zeta):
        self.t = hZeta(zeta, self.training_data)

    def Target(self):
        temp = self.t
        temp = np.nditer(temp, flags=['f_index'])

        loss = 0

        for item in temp:
            loss += np.log(1 + np.exp(-self.y[temp.index]*item))

        return -loss

    def gradient(self, zeta):
        temp = np.zeros_like(zeta)
        t = self.t
        t = np.nditer(t, flags=['f_index'])
        for item in t:
            temp += np.exp(-self.y[t.index]*item) / (1 + np.exp(-self.y[t.index]*item)) * \
                    (-self.y[t.index]*self.training_data[t.index, :].reshape(9, 1))

        return temp

    def Newton(self):
        temp = np.zeros((9, 9))
        t = self.t
        for indexX in range(9):
            for indexY in range(0, indexX+1):
                for item in range(data_num - self.__test):
                    temp[indexX][indexY] += np.exp(-self.y[item]*t[item])*self.training_data[item, :][indexX]*\
                                            self.training_data[item, :][indexY] / \
                                            (1 + np.exp(-self.y[item]*t[item])) ** 2

                temp[indexY][indexX] = temp[indexX][indexY]

        return temp

    def getResult(self, limit):

        for index in range(len(self.y)):
            if self.Y[index] == limit:
                self.y[index] = 1
            else:
                self.y[index] = -1

        print(self.y)

        Zeta = np.zeros((9, 1))
        flag = 1e9
        self.preSolve(Zeta)
        LL = self.Target()
        while abs(flag) > 0.000001:
            Zeta -= np.matmul(np.linalg.inv(self.Newton()), self.gradient(Zeta))
            # Zeta -= 0.001 * self.gradient(Zeta)
            self.preSolve(Zeta)
            flag = LL - self.Target()
            LL = LL - flag

        return Zeta

    def getFeature(self):
        for i in range(5):
            self.feature.append(self.getResult(i))

        self.feature = np.array(self.feature)

    def main(self):
        self.getFeature()
        prediction = hZeta(self.feature[0], self.test_data)
        for i in range(1, 5):
            prediction = np.c_[prediction, hZeta(self.feature[i], self.test_data)]

        predictCategories = np.argmax(prediction, axis=1)

        result = np.c_[self.test_data, self.testY.reshape(self.__test, 1), predictCategories]

        np.savetxt('testNum'+str(self.__test)+'Result.txt', result, fmt='%d')


print('Generating prediction ...')

for key in [1000, 2000, 2960, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
    s = Solve(key)
    s.main()

print('Generation over')





