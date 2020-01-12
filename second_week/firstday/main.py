# _*_ coding:utf-8 _*_
"""
 @author: LaiJinHan
 @time：2019/7/27 21:50
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from DLStudy.L_layer_network import *
from DLStudy.second_week.firstday import init_utils, reg_utils, gc_utils
from DLStudy.second_week.firstday.reg_utils import load_2D_dataset

"""

# 几种不同的初始化参数
def initialize_parameters(layer_dims, type="he"):
    parameters = {}
    L = len(layer_dims)
    for i in range(1, L):
        if type == "zeros":
            parameters["W" + str(i)] = np.zeros((layer_dims[i], layer_dims[i - 1]))
        elif type == "random":
            parameters["W" + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * 4
        else:
            parameters["W" + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * np.sqrt(
                2 / layer_dims[i - 1])

        parameters["b" + str(i)] = np.zeros([layer_dims[i], 1])

        assert (parameters["W" + str(i)].shape == (layer_dims[i], layer_dims[i - 1]))
        assert (parameters["b" + str(i)].shape == (layer_dims[i], 1))
    return parameters

"""


def model_init(X, Y, learning_rate=0.01, num_iterations=15000, print_cost=True, type="he", is_polt=True):
    """
    实现一个三层的神经网络：LINEAR ->RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID

    参数：
        X - 输入的数据，维度为(2, 要训练/测试的数量)
        Y - 标签，【0 | 1】，维度为(1，对应的是输入的数据的标签)
        learning_rate - 学习速率
        num_iterations - 迭代的次数
        print_cost - 是否打印成本值，每迭代1000次打印一次
        type - 字符串类型，初始化的类型【"zeros" | "random" | "he"】
        is_polt - 是否绘制梯度下降的曲线图
    返回
        parameters - 学习后的参数
    """
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 10, 5, 1]

    # 选择初始化参数的类型
    if type == "zeros":
        parameters = initialize_parameters(layers_dims, "zeros")
    elif type == "random":
        parameters = initialize_parameters(layers_dims, "random")
    elif type == "he":
        parameters = initialize_parameters(layers_dims, "he")
    else:
        print("错误的初始化参数！程序退出")
        exit

    # 开始学习
    for i in range(0, num_iterations):
        # 前向传播
        AL, caches = L_model_forward(X, parameters)

        # 计算成本
        cost = compute_cost(AL, Y)

        # 反向传播
        grads = L_model_backward(AL, Y, caches)

        # 更新参数
        parameters = update_parameters(parameters, grads, learning_rate)

        # 记录成本
        if i % 1000 == 0:
            costs.append(cost)
            # 打印成本
            if print_cost:
                print("第" + str(i) + "次迭代，成本值为：" + str(cost))

    # 学习完毕，绘制成本曲线
    if is_polt:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    # 返回学习完毕后的参数
    return parameters,grads


def model_reg_or_dropout(X, Y, learning_rate=0.3, num_iterations=20000, print_cost=True, is_plot=True, lambd=0,
                         keep_prob=1):
    """
    实现一个三层的神经网络：LINEAR ->RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID

    参数：
        X - 输入的数据，维度为(2, 要训练/测试的数量)
        Y - 标签，【0(蓝色) | 1(红色)】，维度为(1，对应的是输入的数据的标签)
        learning_rate - 学习速率
        num_iterations - 迭代的次数
        print_cost - 是否打印成本值，每迭代10000次打印一次，但是每1000次记录一个成本值
        is_polt - 是否绘制梯度下降的曲线图
        lambd - 正则化的超参数，实数,0表示不使用正则化参数
        keep_prob - 随机删除节点的概率,1表示不删除节点
    返回
        parameters - 学习后的参数
    """
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 20, 3, 1]

    # 初始化参数
    parameters = initialize_parameters(layers_dims)

    # 开始学习
    for i in range(0, num_iterations):
        # 前向传播
        ##是否随机删除节点
        if keep_prob == 1:
            ###不随机删除节点
            AL, caches = L_model_forward(X, parameters)
        elif keep_prob < 1:
            ###随机删除节点
            AL, caches = L_model_forward_with_dropout(X, parameters, keep_prob)

        else:
            print("keep_prob参数错误！程序退出。")
            exit


        # 计算成本
        ## 是否使用二范数
        if lambd == 0:
            ###不使用L2正则化
            cost = compute_cost(AL, Y)
        else:
            ###使用L2正则化
            cost = compute_cost_with_reg(AL, Y, parameters, lambd)

        # 反向传播
        ##可以同时使用L2正则化和随机删除节点，但是本次实验不同时使用。
        assert (lambd == 0 or keep_prob == 1)

        ##两个参数的使用情况
        if (lambd == 0 and keep_prob == 1):
            ### 不使用L2正则化和不使用随机删除节点
            grads = L_model_backward(AL, Y, caches)
        elif lambd != 0:
            ### 使用L2正则化，不使用随机删除节点
            grads = L_model_backward_with_reg(AL, Y, caches, lambd)
        elif keep_prob < 1:
            ### 使用随机删除节点，不使用L2正则化
            grads = L_model_backward_with_dropout(AL, Y, caches, keep_prob)



        # 更新参数
        parameters = update_parameters(parameters, grads, learning_rate)

        # 记录并打印成本
        if i % 1000 == 0:
            ## 记录成本
            costs.append(cost)
            if print_cost and i % 10000 == 0:
                # 打印成本
                print("第" + str(i) + "次迭代，成本值为：" + str(cost))

    # 是否绘制成本曲线图
    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (x1,000)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    # 返回学习后的参数
    return parameters



plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

"""
#初始化参数
train_X, train_Y, test_X, test_Y = init_utils.load_dataset(is_plot=False)

parameters = model_init(train_X, train_Y, type = "he",is_polt=True,print_cost=False)
print ("训练集:")
predictions_train = predict(train_X, train_Y, parameters)
print ("测试集:")
predictions_test = predict(test_X, test_Y, parameters)

plt.title("Model with large random initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
init_utils.plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, np.squeeze(train_Y))
"""

"""

train_X, train_Y, test_X, test_Y = load_2D_dataset(is_plot=False)

parameters = model_reg_or_dropout(train_X, train_Y, keep_prob=0.86, learning_rate=0.03,is_plot=True)

print("使用随机删除节点，训练集:")
predictions_train = predict(train_X, train_Y, parameters)
print("使用随机删除节点，测试集:")
predictions_test = predict(test_X, test_Y, parameters)
plt.title("Model with dropout")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, np.squeeze(train_Y))
"""
