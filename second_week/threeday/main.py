# _*_ coding:utf-8 _*_
"""
 @author: LaiJinHan
 @time：2019/8/2 19:15
"""
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import time

from DLStudy.second_week.threeday import tf_utils


def one_hot_matrix(lables):
    C = np.max(lables) + 1
    m = lables.shape[1]
    one_hot = np.zeros((C, m))
    one_hot[lables, range(m)] = 1
    # one_hot= np.eye(C)[lables.reshape(-1)].T
    return one_hot


def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")
    return X, Y


def initialize_parameters():
    """
    维度为[12288,25,12,6]
    """
    tf.set_random_seed(1)  # 指定随机种子
    layer_dimes = [12288, 25, 12, 6]
    parameters = {}
    l = len(layer_dimes)
    for i in range(1, l):
        parameters["W" + str(i)] = tf.get_variable("W" + str(i), [layer_dimes[i], layer_dimes[i - 1]],
                                                   initializer=tf.contrib.layers.xavier_initializer(seed=1))
        parameters["b" + str(i)] = tf.get_variable("b" + str(i), [layer_dimes[i], 1],
                                                   initializer=tf.zeros_initializer())
    return parameters


def forward_propagation(X, parameters):
    """
    LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    :param X: 
    :param parameters: 
    :return: 
    """
    l = len(parameters) // 2
    A = X
    for i in range(1, l):
        A_prev = A
        A = tf.nn.relu(tf.add(tf.matmul(parameters["W" + str(i)], A_prev), parameters["b" + str(i)]))
    ZL = tf.add(tf.matmul(parameters["W" + str(l)], A), parameters["b" + str(l)])
    return ZL


def compute_cost(ZL, Y):
    # Y为（6，32）
    logits = tf.transpose(ZL)  # 转置（？，6）
    labels = tf.transpose(Y)
    x = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cost = tf.reduce_mean(x)
    # softmax_cross_entropy_with_logits计算交叉熵函数，注意的是logits是未使用softmax的
    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001, num_epochs=1500, minibatch_size=32, print_cost=True,
          is_plot=True):
    """
       实现一个三层的TensorFlow神经网络：LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX
       参数：
           X_train - 训练集，维度为（输入大小（输入节点数量） = 12288, 样本数量 = 1080）
           Y_train - 训练集分类数量，维度为（输出大小(输出节点数量) = 6, 样本数量 = 1080）
           X_test - 测试集，维度为（输入大小（输入节点数量） = 12288, 样本数量 = 120）
           Y_test - 测试集分类数量，维度为（输出大小(输出节点数量) = 6, 样本数量 = 120）
       返回：
           parameters - 学习后的参数

    """
    ops.reset_default_graph()  # 能够重新运行模型而不覆盖tf变量
    tf.set_random_seed(1)
    seed = 3
    n_x, m = X_train.shape  # 获取输入节点数量和样本数
    n_y = Y_train.shape[0]  # 获取输出节点数量
    costs = []  # 成本集
    # 给X和Y创建placeholder
    X, Y = create_placeholders(n_x, n_y)
    # 初始化参数
    parameters = initialize_parameters()
    # 前向传播
    ZL = forward_propagation(X, parameters)
    # 计算成本
    cost = compute_cost(ZL, Y)
    # 反向传播，使用Adam优化
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # 初始化所有的变量
    init = tf.global_variables_initializer()

    # 开始会话并计算
    with tf.Session() as sess:
        # 初始化
        sess.run(init)
        # 正常训练的循环
        for epoch in range(num_epochs):
            epoch_cost = 0  # 每代的成本
            num_minibatches = int(m / minibatch_size)  # minibatch的总数量
            seed = seed + 1
            minibatches = tf_utils.random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                # 选择一个minibatch
                minibatch_X, minibatch_Y = minibatch

                # 数据已经准备好了，开始运行session
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                # 计算这个minibatch在这一代中所占的误差
                epoch_cost = epoch_cost + minibatch_cost / num_minibatches

            # 记录并打印成本
            ## 记录成本
            if epoch % 5 == 0:
                costs.append(epoch_cost)
                # 是否打印：
                if print_cost and epoch % 100 == 0:
                    print("epoch = " + str(epoch) + "    epoch_cost = " + str(epoch_cost))

        # 是否绘制图谱
        if is_plot:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

        # 保存学习后的参数
        parameters = sess.run(parameters)
        print("参数已经保存到session。")

        # 计算当前的预测结果
        correct_prediction = tf.equal(tf.argmax(ZL, 0), tf.argmax(Y))  # tf.equal(a,b)判断两元素是否相等
        # tf.argmax(x,axis)找出行或者列最大值的索引

        # 计算准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # tf.reduce_mean计算指定维度的平均值
        # tf.cast转换数据格式

        # eval()是启动计算的另外一种方法
        print("训练集的准确率：", accuracy.eval({X: X_train, Y: Y_train}))
        print("测试集的准确率:", accuracy.eval({X: X_test, Y: Y_test}))

    return parameters


def predict(X, parameters):
    l = len(parameters) // 2
    par = {}
    for i in range(1, l + 1):
        par["W" + str(i)] = tf.convert_to_tensor(parameters["W" + str(i)])
        par["b" + str(i)] = tf.convert_to_tensor(parameters["b" + str(i)])

    x = tf.placeholder(tf.float32, X.shape)#X.shape(12288,1)
    zl = forward_propagation(x, par)
    p = tf.argmax(zl)

    sess = tf.Session()
    prediction = sess.run(p, feed_dict={x: X})

    return prediction


# labels = np.array([1, 2, 3, 0, 2, 1])
#
# one_hot = one_hot_matrix(labels.reshape(1, -1))
# print(str(one_hot))

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = tf_utils.load_dataset()

X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0],
                                       -1).T  # X_train_orig(1080,64,64,1),Y_train_orig(1,1080)
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

# 归一化数据
X_train = X_train_flatten / 255
X_test = X_test_flatten / 255

Y_train = one_hot_matrix(Y_train_orig)
Y_test = one_hot_matrix(Y_test_orig)
# 开始时间
start_time = time.clock()
# 开始训练
parameters = model(X_train, Y_train, X_test, Y_test)
# 结束时间
end_time = time.clock()
# 计算时差
print("CPU的执行时间 = " + str(end_time - start_time) + " 秒")

# my_image1 = "5.png"  # 定义图片名称
# fileName1 = "images/" + my_image1  # 图片地址
# image1 = mpimg.imread(fileName1)  # 读取图片
# if image1.shape[2] == 4:
#     im = Image.open(fileName1).convert("RGB")
#
# print(image1.shape)
# plt.imshow(image1)
# plt.show()
# # 显示图片
# my_imag = (image1.reshape(1, -1).T) / 255
# print(my_imag.shape)
# # my_image1 = image1.reshape(1, 64 * 64 * 3).T  # 重构图片
# my_image_prediction = tf_utils.predict(my_image1, parameters)  # 开始预测
# print("预测结果: y = " + str(np.squeeze(my_image_prediction)))

"""
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = tf_utils.load_dataset()

X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0],
                                       -1).T  # X_train_orig(1080,64,64,1),Y_train_orig(1,1080)
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

# 归一化数据
X_train = X_train_flatten / 255
X_test = X_test_flatten / 255

# Y_train = tf_utils.convert_to_one_hot(Y_train_orig, 6) #(6,108)
Y_train = one_hot_matrix(Y_train_orig.reshape(-1), 6)
Y_test = one_hot_matrix(Y_test_orig.reshape(-1), 6)

# index = 11
# plt.imshow(X_train_orig[index])
# plt.show()
# print("Y = " + str(np.squeeze(Y_train_orig[:, index])))
print("训练集样本数 = " + str(X_train.shape[1]))
print("测试集样本数 = " + str(X_test.shape[1]))
print("X_train.shape: " + str(X_train.shape))
print("Y_train.shape: " + str(Y_train.shape))
print("X_test.shape: " + str(X_test.shape))
print("Y_test.shape: " + str(Y_test.shape))
"""
