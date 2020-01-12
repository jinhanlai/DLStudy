# _*_ coding:utf-8 _*_
"""
 @author: LaiJinHan
 @time：2019/7/25 12:18
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage

import DLStudy.L_layer_network


def sigmod(Z):
    A = 1.0 / (1 + 1 / np.exp(Z))
    return A


def sigmod_backward(dA, Z):
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)
    return dZ


def relu(Z):
    A = np.maximum(0, Z)
    assert (A.shape == Z.shape)
    return A


def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # print(dA.shape,dZ.shape,Z.shape)
    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


# 初始化n层神经网络参数，layers_dims包含每层的节点数目
def initialize_parameters(layers_dims):
    # layers_dims - 包含我们网络中每个图层的节点数量的列表
    # np.random.seed(3)

    l = len(layers_dims)  # 加输入层一共5层

    parameters = {}
    for i in range(1, l):
        # 提问*0.01 与 / np.sqrt(layers_dims[i - 1]) 的区别，为什么要这样设置
        # 如果*0.01最后的成本值更新很慢，趋近0.64；而/ np.sqrt(layers_dims[i - 1])学习效率很好  why？？
        # parameters["W" + str(i)] = np.random.randn(layers_dims[i], layers_dims[i - 1]) * 0.01
        parameters["W" + str(i)] = np.random.randn(layers_dims[i], layers_dims[i - 1]) / np.sqrt(layers_dims[i - 1])
        parameters["b" + str(i)] = np.zeros((layers_dims[i], 1))
        assert (parameters["W" + str(i)].shape == (layers_dims[i], layers_dims[i - 1]))
        assert (parameters["b" + str(i)].shape == (layers_dims[i], 1))
    return parameters


def linear_forward(A_prew, W, b):
    Z = np.dot(W, A_prew) + b
    assert (Z.shape == (W.shape[0], A_prew.shape[1]))
    return Z


# cache由(A_prew,W,b,Z) 其实Z=np.dot(W,A_prew)+b
def activation_forward_propagation(A_prew, W, b, active_type="sigmoid"):
    if active_type == "sigmoid":
        Z = linear_forward(A_prew, W, b)
        A = sigmod(Z)
    elif active_type == "relu":
        Z = linear_forward(A_prew, W, b)
        A = relu(Z)

    assert (A.shape == (W.shape[0], A_prew.shape[1]))
    cache = (A_prew, W, b, Z)

    return A, cache

#X为最开始输入的数据
def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2  # 整除  因为parameters是有w和b组成的，所以长度要除以2，值为4
    for l in range(1, L):
        A_prev = A  # (12288,2009)
        A, cache = activation_forward_propagation(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)

    AL, cache = activation_forward_propagation(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches


def compute_cost(AL, Y):
    m = Y.shape[1]

    cost = -np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), 1 - Y)) / m

    cost = float(np.squeeze(cost))
    assert (isinstance(cost, float))

    return cost


# 第L层的反向传播
def linear_backward(dZ, A_prev, W, b):
    """
    参数：
         dZ - 相对于（当前第l层的）线性输出的成本梯度
         A_prev：前一层的输出

    返回：
         dA_prev - 相对于激活（前一层l-1）的成本梯度，与A_prev维度相同
         dW - 相对于W（当前层l）的成本梯度，与W的维度相同
         db - 相对于b（当前层l）的成本梯度，与b维度相同
    """
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def activation_backward_propagation(dA, cache, activation="relu"):
    A_prew, W, b, Z = cache

    if activation == "relu":
        dZ = relu_backward(dA, Z)
        dA_prev, dW, db = linear_backward(dZ, A_prew, W, b)

    elif activation == "sigmoid":
        dZ = sigmod_backward(dA, Z)
        dA_prev, dW, db = linear_backward(dZ, A_prew, W, b)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)

    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    grads["dA" + str(L)] = dAL

    current_cache = caches[L - 1]
    dA_prev, dW, db = activation_backward_propagation(dAL, current_cache, activation="sigmoid")
    # print(dA_prev.shape, dW.shape, db.shape)
    grads["dA" + str(L - 1)] = dA_prev
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db

    # print(grads)
    for l in reversed(range(L - 1)):  # 2、1、0,依次计算第二层，第一层，输入层
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = activation_backward_propagation(grads["dA" + str(l + 1)], current_cache,
                                                                         "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # 整除 L=4

    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False, isPlot=True):
    """
        print_cost - 是否打印成本值，每100次打印一次
        isPlot - 是否绘制出误差值的图谱

    返回：
     parameters - 模型学习的参数。 然后他们可以用来预测。
    """
    # np.random.seed(1)
    costs = []

    parameters = initialize_parameters(layers_dims)

    for i in range(0, num_iterations):

        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)

        grads = L_model_backward(AL, Y, caches)

        parameters = update_parameters(parameters, grads, learning_rate)

        # 打印成本值，如果print_cost=False则忽略
        if i % 100 == 0:
            # 记录成本
            costs.append(cost)
            # 是否打印成本值
            if print_cost:
                print("第", i, "次迭代，成本值为：", np.squeeze(cost))

    # 迭代完成，根据条件绘制图
    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    return parameters


def predict(X, y, parameters):
    """
    该函数用于预测L层神经网络的结果，当然也包含两层

    参数：
     X - 测试集
     y - 标签
     parameters - 训练模型的参数

    返回：
     p - 给定数据集X的预测
    """

    m = X.shape[1]
    n = len(parameters) // 2  # 神经网络的层数
    p = np.zeros((1, m))

    # 根据参数前向传播
    AL, caches = L_model_forward(X, parameters)

    for i in range(0, AL.shape[1]):
        if AL[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print("准确度为: " + str(float(np.sum((p == y)) / m)))

    return p


# p(1,50) x(12288,50) Y(1,50)
def print_mislabeled_images(classes, X, y, p):
    """
    绘制预测和实际不同的图像。
        X - 数据集
        y - 实际的标签
        p - 预测
    """
    a = p + y  # (1,50)
    # np.where(a == 1)返回满足条件的索引
    mislabeled_indices = np.asarray(np.where(a == 1))  # (2,11) #第一行代表在原矩阵中的行数，第二行返回满足条件的列数

    plt.rcParams['figure.figsize'] = (40.0, 40.0)  # set default size of plots
    num_images = len(mislabeled_indices[0])  # 11
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        plt.title(
            "Prediction: " + classes[int(p[0, index])].decode("utf-8") + " \n Class: " + classes[y[0, index]].decode(
                "utf-8"))

    plt.show()


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()

train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255
train_y = train_set_y
test_x = test_x_flatten / 255
test_y = test_set_y
layers_dims = [12288, 20, 7, 5, 1]  # 5-layer model

parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=False, isPlot=True)

pred_train = predict(train_x, train_y, parameters)  # 训练集
pred_test = predict(test_x, test_y, parameters)  # 测试集

print_mislabeled_images(classes, test_x, test_y, pred_test)  # pred_test=(1,m)


"""
my_image = "my_image.png"  # change this to the name of your image file
my_label_y = [1]  # the true class of your image (1 -> cat, 0 -> non-cat)
# END CODE HERE ##

fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(64,64)).reshape((64*64*3,1))
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")

"""
