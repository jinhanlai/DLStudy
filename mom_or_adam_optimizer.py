# _*_ coding:utf-8 _*_
"""
 @author: LaiJinHan
 @time：2019/8/1 22:05
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from DLStudy.L_layer_network import *

def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    # 第一步打乱输入集数据顺序
    permutation = list(np.random.permutation(m))  # 返回长度为m的随机数组，而且里面的数目是0到m-1
    shuffled_x = X[:, permutation]  # 将每一列的数据按照permutation的顺序来重新排列
    shuffled_y = Y[:, permutation].reshape((1, m))
    # 第二步分割数据集数据
    num_all_mini_batches = math.floor(m / mini_batch_size)  # 取整数部分，向上取整(math.floor(2.9)=2),math.ceil向下取整
    for i in range(0, num_all_mini_batches):
        mini_batch_X = shuffled_x[:, i * mini_batch_size:(i + 1) * mini_batch_size]
        mini_batch_Y = shuffled_y[:, i * mini_batch_size:(i + 1) * mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_x[:, num_all_mini_batches * mini_batch_size:]
        mini_batch_Y = shuffled_y[:, num_all_mini_batches * mini_batch_size:]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def initialize_velocity(parameters):
    L = len(parameters) // 2
    v = {}
    for i in range(0, L):
        v["dW" + str(i + 1)] = np.zeros_like(parameters["W" + str(i + 1)])
        v["db" + str(i + 1)] = np.zeros_like(parameters["b" + str(i + 1)])

    return v


def update_parameters_with_momentun(parameters, grads, v, learning_rate, beta=0.9):
    L = len(parameters) // 2
    for i in range(0, L):
        v["dW" + str(i + 1)] = beta * v["dW" + str(i + 1)] + (1 - beta) * grads["dW" + str(i + 1)]
        v["db" + str(i + 1)] = beta * v["db" + str(i + 1)] + (1 - beta) * grads["db" + str(i + 1)]
        parameters["W" + str(i + 1)] = parameters["W" + str(i + 1)] - learning_rate * v["dW" + str(i + 1)]
        parameters["b" + str(i + 1)] = parameters["b" + str(i + 1)] - learning_rate * v["db" + str(i + 1)]

    return parameters, v


def initialize_adam(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}
    for i in range(0, L):
        v["dW" + str(i + 1)] = np.zeros_like(parameters["W" + str(i + 1)])
        v["db" + str(i + 1)] = np.zeros_like(parameters["b" + str(i + 1)])
        s["dW" + str(i + 1)] = np.zeros_like(parameters["W" + str(i + 1)])
        s["db" + str(i + 1)] = np.zeros_like(parameters["b" + str(i + 1)])

    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    l = len(parameters) // 2
    v_correct = {}  # 偏差修正后的值
    s_correct = {}
    for i in range(0, l):
        v["dW" + str(i + 1)] = beta1 * v["dW" + str(i + 1)] + (1 - beta1) * grads["dW" + str(i + 1)]
        v["db" + str(i + 1)] = beta1 * v["db" + str(i + 1)] + (1 - beta1) * grads["db" + str(i + 1)]

        v_correct["dW" + str(i + 1)] = v["dW" + str(i + 1)] / (1 - np.power(beta1, t))
        v_correct["db" + str(i + 1)] = v["db" + str(i + 1)] / (1 - np.power(beta1, t))

        s["dW" + str(i + 1)] = beta2 * s["dW" + str(i + 1)] + (1 - beta2) * np.square(grads["dW" + str(i + 1)])
        s["db" + str(i + 1)] = beta2 * s["db" + str(i + 1)] + (1 - beta2) * np.square(grads["db" + str(i + 1)])

        s_correct["dW" + str(i + 1)] = s["dW" + str(i + 1)] / (1 - np.power(beta2, t))
        s_correct["db" + str(i + 1)] = s["db" + str(i + 1)] / (1 - np.power(beta2, t))

        parameters["W" + str(i + 1)] = parameters["W" + str(i + 1)] - learning_rate * (
            v_correct["dW" + str(i + 1)] / (np.sqrt(s_correct["dW" + str(i + 1)] + epsilon)))
        parameters["b" + str(i + 1)] = parameters["b" + str(i + 1)] - learning_rate * (
            v_correct["db" + str(i + 1)] / (np.sqrt(s_correct["db" + str(i + 1)] + epsilon)))

    return parameters, v, s


def model_optimizer(X, Y, layers_dims, optimizer, learning_rate=0.0007, mini_batch_size=64, beta=0.9, beta1=0.9,
                    beta2=0.999, epsilon=1e-8, num_iteration=10000, print_cost=True, is_plot=True):
    l = len(layers_dims)
    costs = []
    t = 0
    seed = 10
    parameters = initialize_parameters(layers_dims)

    if optimizer == "gd":
        pass  # 不使用任何优化器，直接使用梯度下降法
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)
    else:
        print("optimizer参数错误，程序退出。")
        exit(1)

    # 开始学习
    for i in range(0, num_iteration):
        seed = seed + 1
        mini_batches = random_mini_batches(X, Y, mini_batch_size, seed)

        for mini_batch_i in mini_batches:

            mini_batch_X, mini_batch_Y = mini_batch_i

            AL, caches = L_model_forward(mini_batch_X, parameters)

            cost = compute_cost(AL, mini_batch_Y)

            grads = L_model_backward(AL, mini_batch_Y, caches)

            if optimizer == "gd":
                parameters = update_parameters(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentun(parameters, grads, v, learning_rate, beta)
            elif optimizer == "adam":
                t = t + 1
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2,
                                                               epsilon)

        if i % 100 == 0:
            costs.append(cost)
            # 是否打印误差值
            if print_cost and i % 1000 == 0:
                print("第" + str(i) + "次遍历整个数据集，当前误差值：" + str(cost))

    # 是否绘制曲线图
    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('epochs (per 100)')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()

    return parameters
