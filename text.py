import numpy as np


def one_hot_matrix(lables):
    C = np.max(lables) + 1
    m = lables.shape[1]
    one_hot = np.zeros((C, m))
    one_hot[lables, range(m)] = 1
    # one_hot= np.eye(C)[lables.reshape(-1)].T
    return one_hot


def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), 0)


def compute_cost_with_softmax(AL, Y):
    m = Y.shape[1]
    cost = -np.sum(np.multiply(Y, np.log(AL))) / m
    # cost = -np.sum(np.multiply(np.log(np.exp(AL) / np.sum(np.exp(AL), 0)),Y))/m

    return cost


Z = np.array([[2, 1, 3, 6, 7], [3, 3, 5, 7, 8], [5, 4, 6, 9, 10], [6, 3, 7, 10, 11]])
Y = np.array([[1, 2, 0, 3, 2]])
Y1 = one_hot_matrix(Y)
A = softmax(Z)

print(A)
print(Y1)
print(compute_cost_with_softmax(A, Y1))
# print(LossFunc(Y, Z))
# print(A)
# print(compute_cost_with_softmax(A, Y))
# print(softmaxwithloss(A, Y))
