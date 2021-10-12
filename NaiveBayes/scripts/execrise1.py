import numpy as np

x = np.array([[1, 1, 1, 1, 1], [2005, 2006, 2007, 2008, 2009]])
y = np.array([12, 19, 29, 37, 45]).reshape(5, 1)

zeta = np.matmul(np.matmul(np.linalg.inv(np.matmul(x, x.T)), x), y)

print(np.dot(zeta.T, np.array([1, 2012])))