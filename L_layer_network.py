# _*_ coding:utf-8 _*_
"""
 @author: LaiJinHan
 @time：2019/7/28 20:13
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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


def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))


def change_image_4_to_3_channels(image, image_path):
    """
    path='D:\\PythonProject\\DLStudy\second_week\\threeday\\images\\5.png'
    img = Image.open(path)
    new_image = change_image_4_to_3_channels(img, path)
    
    image1 = mpimg.imread('D:\\PythonProject\\DLStudy\second_week\\threeday\\images\\2.png')
    print(image1.shape)
    """
    # 4通道转3通道
    if image.mode == 'RGBA':
        r, g, b, a = image.split()
        image = Image.merge("RGB", (r, g, b))
        image.save(image_path)
    #1通道变3通道
    elif image.mode != 'RGB':
        image = image.convert("RGB")
        image.save(image_path)
    return image


def one_hot_matrix(lables):
    C = np.max(lables) + 1
    m = lables.shape[1]
    one_hot = np.zeros((C, m))
    one_hot[lables, range(m)] = 1
    # one_hot= np.eye(C)[lables.reshape(-1)].T
    return one_hot


def initialize_parameters(layer_dims, type="he"):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    for i in range(1, L):
        if type == "zeros":
            parameters["W" + str(i)] = np.zeros((layer_dims[i], layer_dims[i - 1]))
        elif type == "random":
            parameters["W" + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * 10
        else:
            parameters["W" + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * np.sqrt(
                2 / layer_dims[i - 1])

        parameters["b" + str(i)] = np.zeros([layer_dims[i], 1])
        assert (parameters["W" + str(i)].shape == (layer_dims[i], layer_dims[i - 1]))
        assert (parameters["b" + str(i)].shape == (layer_dims[i], 1))
    return parameters


def linear_forward(A_prew, W, b):
    # print(A_prew.shape,W.shape)
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


# X为最开始输入的数据
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


def L_model_forward_with_dropout(X, parameters, keep_prob=0.5):
    """
       实现具有随机舍弃节点的前向传播。
       LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.

       参数：
           X  - 输入数据集，维度为（2，示例数）
           parameters - 包含参数“W1”，“b1”，“W2”，“b2”，“W3”，“b3”的python字典：
               W1  - 权重矩阵，维度为（20,2）
               b1  - 偏向量，维度为（20,1）
               W2  - 权重矩阵，维度为（3,20）
               b2  - 偏向量，维度为（3,1）
               W3  - 权重矩阵，维度为（1,3）
               b3  - 偏向量，维度为（1,1）
           keep_prob  - 随机删除的概率，实数
       返回：
           AL  - 最后的激活值，维度为（1,1），正向传播的输出
           caches - 存储了一些用于计算反向传播的数值的元组
       """

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID

    # # 下面的步骤1-4对应于上述的步骤1-4。
    # D1 = np.random.rand(A1.shape[0], A1.shape[1])  # 步骤1：初始化矩阵D1 = np.random.rand(..., ...)
    # D1 = D1 < keep_prob  # 步骤2：将D1的值转换为0或1（使​​用keep_prob作为阈值）
    # A1 = A1 * D1  # 步骤3：舍弃A1的一些节点（将它的值变为0或False）
    # A1 = A1 / keep_prob  # 步骤4：缩放未舍弃的节点(不为0)的值

    np.random.seed(1)

    caches = []
    A = X
    L = len(parameters) // 2  # 整除  因为parameters是有w和b组成的，所以长度要除以2，值为4
    for l in range(1, L):
        A_prev = A  # (12288,2009)
        A, cache = activation_forward_propagation(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        D1 = np.random.rand(A.shape[0], A.shape[1])
        D1 = D1 < keep_prob
        A *= D1
        A /= keep_prob
        cache = (cache, D1)
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


def compute_cost_with_reg(AL, Y, parameters, lambd):
    """
    实现公式2的L2正则化计算成本

    参数：
        AL - 正向传播的输出结果，维度为（输出节点数量，训练/测试的数量）
        Y - 标签向量，与数据一一对应，维度为(输出节点数量，训练/测试的数量)
        parameters - 包含模型学习后的参数的字典
    返回：
        cost - 使用公式2计算出来的正则化损失的值

    """

    m = Y.shape[1]

    # 加输入层一共4层
    W_square_sum = 0
    for i in range(1, (len(parameters) // 2 + 1)):
        Wi = parameters["W" + str(i)]
        W_square_sum += np.sum(np.square(Wi))

    cross_entropy_cost = compute_cost(AL, Y)

    L2_reg_cost = (lambd / (2 * m)) * (W_square_sum)
    cost = cross_entropy_cost + L2_reg_cost
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


def L_model_backward_with_reg(AL, Y, caches, lambd):
    """
      实现我们添加了L2正则化的模型的后向传播。

      参数：
          X - 输入数据集，维度为（输入节点数量，数据集里面的数量）
          Y - 标签，维度为（输出节点数量，数据集里面的数量）
          caches - 来自L_model_forward()的caches输出
          lambd - regularization超参数，实数

      返回：
          grads - 一个包含了每个参数、激活值和预激活值变量的梯度的字典
      """

    grads = {}
    L = len(caches)

    m = AL.shape[1]

    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    grads["dA" + str(L)] = dAL

    current_cache = caches[L - 1]
    dA_prev, dW, db = activation_backward_propagation(dAL, current_cache, activation="sigmoid")

    dW += ((lambd * current_cache[1]) / m)  # current_cache=(A_prev,W,b,Z)
    # print(dA_prev.shape, dW.shape, db.shape)
    grads["dA" + str(L - 1)] = dA_prev
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db

    # print(grads)
    for l in reversed(range(L - 1)):  # 2、1、0,依次计算第二层，第一层，输入层
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = activation_backward_propagation(grads["dA" + str(l + 1)], current_cache,
                                                                         "relu")
        dW_temp += ((lambd * current_cache[1]) / m)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def L_model_backward_with_dropout(AL, Y, caches, keep_prob):
    """
    实现我们随机删除的模型的后向传播。
    参数：
        X  - 输入数据集，维度为（2，示例数）
        Y  - 标签，维度为（输出节点数量，示例数量）
        cache - 来自forward_propagation_with_dropout（）的cache输出
        keep_prob  - 随机删除的概率，实数

    返回：
        grads - 一个关于每个参数、激活值和预激活变量的梯度值的字典
    """
    grads = {}
    L = len(caches)

    m = AL.shape[1]

    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    grads["dA" + str(L)] = dAL

    current_cache = caches[L - 1]  # 注意caches中的第一个和最后一个没有D1
    dA_prev, dW, db = activation_backward_propagation(dAL, current_cache, activation="sigmoid")

    D1 = caches[L - 2][1]
    dA_prev *= D1
    dA_prev /= keep_prob

    # print(dA_prev.shape, dW.shape, db.shape)
    grads["dA" + str(L - 1)] = dA_prev
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db

    # print(grads)
    for l in reversed(range(1, L - 1)):  # 计算隐含层
        current_cache = caches[l]  # cache=(cache,D1) 其中括号里面的cache=(A_prew,W,b,Z)

        dA_prev_temp, dW_temp, db_temp = activation_backward_propagation(grads["dA" + str(l + 1)], current_cache[0],
                                                                         "relu")
        D1 = caches[l - 1][1]
        dA_prev_temp *= D1
        dA_prev_temp /= keep_prob
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    current_cache = caches[0]  # 计算输入层
    dA_prev_temp, dW_temp, db_temp = activation_backward_propagation(grads["dA" + str(1)], current_cache[0],
                                                                     "relu")
    grads["dA" + str(0)] = dA_prev_temp
    grads["dW" + str(1)] = dW_temp
    grads["db" + str(1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # 整除 L=4

    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
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


def predict_dec(parameters, X):
    """
    Used for plotting decision boundary.

    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (m, K)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    # Predict using forward propagation and a classification threshold of 0.5
    AL, cache = L_model_forward(X, parameters)
    predictions = (AL > 0.5)
    return predictions


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


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()
