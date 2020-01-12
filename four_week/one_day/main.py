# _*_ coding:utf-8 _*_
"""
 @author: LaiJinHan
 @time：2019/8/7 10:28
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py


def zero_pad(X, pad):
    """
    :param X: 输入的数据集,维度为（样本数，图像高度，图像宽度，图像通道数）
    :param pad: 填充量
    :return: 
    """
    X_pad = np.pad(X, (
        (0, 0),  # 样本数不填充
        (pad, pad),  # 图像高度，上下个填充pad行
        (pad, pad),  # 图像宽度，左右各填充pad列
        (0, 0)),  # 通道数不填充
                   'constant', constant_values=0)  # 连续一样的填充0
    return X_pad


def conv_single_step(a_slice_prev, W, b):
    """
    :param a_slice_prev: 输入数据的一个片段(过滤器大小，过滤器大小，上一个通道数)
    :param W: 过滤器，权重参数（过滤器大小，过滤器大小，上一个通道数）
    :param b: 偏置（1,1,1）
    :return: 
    """
    s = np.multiply(a_slice_prev, W) + b
    Z = np.sum(s)
    return Z


def conv_forward(A_prev, W, b, hparameters):
    """
    :param A_prev: 上一层的激活输出矩阵(m,n_H_prev,n_W_prev,n_C_prev),（样本数量，上一层图像的高度，上一层图像的宽度，上一层过滤器数量）
    :param W: 权重矩阵，维度为(f, f, n_C_prev, n_C)，（过滤器大小，过滤器大小，上一层的过滤器数量，这一层的过滤器数量）
    :param b: 维度为(1, 1, 1, n_C)，（1,1,1,这一层的过滤器数量）
    :param hparameters:  包含了"stride"与 "pad"的超参数字典。
    :return: Z:卷积输出（样本数量，图像的高度，图像的宽度，过滤器数量）
            cache - 缓存了一些反向传播函数conv_backward()需要的一些数据
    """
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    f, f, n_C_prev, n_C = W.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    n_H = int((n_H_prev + 2 * pad - f) / stride) + 1  # 计算卷积过后图像的宽度和高度
    n_W = int((n_W_prev + 2 * pad - f) / stride) + 1
    Z = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = zero_pad(A_prev, pad)  # 填充输入的矩阵

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    v_start = h * stride  # 垂直方向
                    v_end = v_start + f
                    h_start = w * stride  # 水平方向
                    h_end = h_start + f
                    a_slice_prew = a_prev_pad[v_start:v_end, h_start:h_end, :]
                    Z[i, h, w, c] = conv_single_step(a_slice_prew, W[:, :, :, c], b[0, 0, 0, c])

    assert (Z.shape == (m, n_H, n_W, n_C))
    cache = (A_prev, W, b, hparameters)
    return Z, cache


def conv_backward(dZ, cache):
    """
    :param dZ: 卷积层的输出Z的 梯度，维度为(m, n_H, n_W, n_C) 
    :param cache: 反向传播所需要的参数，conv_forward()的输出之一
    :return: dA_prev - 卷积层的输入（A_prev）的梯度值，维度为(m, n_H_prev, n_W_prev, n_C_prev)
        dW - 卷积层的权值的梯度，维度为(f,f,n_C_prev,n_C)
        db - 卷积层的偏置的梯度，维度为（1,1,1,n_C）
    """
    A_prev, W, b, hparameters = cache
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dZ.shape
    f, f, n_C_prev, n_C = W.shape
    pad = hparameters["pad"]
    stride = hparameters["stride"]

    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):
        # 选择第i个扩充了的数据的样本,降了一维。
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # 定位切片位置
                    v_start = h * stride  # 垂直方向
                    v_end = v_start + f
                    h_start = w * stride  # 水平方向
                    h_end = h_start + f

                    # 定位完毕，开始切片
                    a_slice = a_prev_pad[v_start:v_end, h_start:h_end, :]

                    # 切片完毕，使用上面的公式计算梯度
                    da_prev_pad[v_start:v_end, h_start:h_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]
        # 设置第i个样本最终的dA_prev,即把非填充的数据取出来。
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
    assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    return dA_prev, dW, db


def pool_forward(A_prev, hparameters, type="max"):
    """
    :param A_prev: 输入数据，维度为(m, n_H_prev, n_W_prev, n_C_prev)
    :param hparameters: 包含了 "f" 和 "stride"的超参数字典
    :param type: 
    :return: A - 池化层的输出，维度为 (m, n_H, n_W, n_C)
             cache - 存储了一些反向传播需要用到的值，包含了输入和超参数的字典。
    """
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    f = hparameters["f"]
    stride = hparameters["stride"]
    # 计算输出维度
    n_H = int((n_H_prev - f) / stride) + 1
    n_W = int((n_W_prev - f) / stride) + 1
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    v_start = h * stride  # 垂直方向
                    v_end = v_start + f
                    h_start = w * stride  # 水平方向
                    h_end = h_start + f

                    a_slice_prev = A_prev[i, v_start:v_end, h_start:h_end, c]
                    if type == "max":
                        A[i, h, w, c] = np.max(a_slice_prev)
                    elif type == "mean":
                        A[i, h, w, c] = np.mean(a_slice_prev)
    assert (A.shape == (m, n_H, n_W, n_C))
    cache = (A_prev, hparameters)
    return A, cache


def create_mask_from_window(x):
    """
    :param x: 一个维度为(f,f)的矩阵
    :return:  - mask包含x的最大值的位置的矩阵(mask跟x有相同的维度，在最大值位置为)，其他为0)
    """
    mask = x == np.max(x)
    return mask


def distribute_value(dz, shape):
    """
    给定一个值，为按矩阵大小平均分配到每一个矩阵位置中。
    :param dz: 输入的实数
    :param shape: 元组，两个值，分别为n_H , n_W
    :return: a - 已经分配好了值的矩阵，里面的值全部一样。
    """
    # 获取矩阵的大小
    (n_H, n_W) = shape
    # 计算平均值
    average = dz / (n_H * n_W)
    # 填充入矩阵
    a = np.ones(shape) * average
    return a


def pool_backward(dA, cache, type="max"):
    """
    
    :param dA: 池化层的输出的梯度，和池化层的输出的维度一样
    :param cache: 池化层前向传播时所存储的参数。
    :param mode: 模式选择，【"max" | "average"】
    :return: dA_prev - 池化层的输入的梯度，和A_prev的维度相同
    """
    (A_prev, hparaeters) = cache
    f = hparaeters["f"]
    stride = hparaeters["stride"]
    (m, n_H, n_W, n_C) = dA.shape

    dA_prev = np.zeros_like(A_prev)
    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # 定位切片位置
                    v_start = h * stride
                    v_end = v_start + f
                    h_start = w * stride
                    h_end = h_start + f

                    if type == "max":
                        a_prev_slice = a_prev[v_start:v_end, h_start:h_end, c]
                        mask = create_mask_from_window(a_prev_slice)
                        dA_prev[i, v_start:v_end, h_start:h_end, c] += np.multiply(mask, dA[i, h, w, c])

                    elif type == "mean":
                        da = dA[i, h, w, c]
                        # 定义过滤器大小
                        shape = (f, f)
                        # 平均分配
                        dA_prev[i, v_start:v_end, h_start:h_end, c] += distribute_value(da, shape)

    assert (dA_prev.shape == A_prev.shape)
    return dA_prev


def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels
    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels
    classes = np.array(test_dataset["list_classes"][:])  # the list of classes
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def one_hot_matrix(lables):
    C = np.max(lables) + 1
    m = lables.shape[1]
    one_hot = np.zeros((C, m))
    one_hot[lables, range(m)] = 1
    # one_hot= np.eye(C)[lables.reshape(-1)].T
    return one_hot


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.
Y_train = one_hot_matrix(Y_train_orig).T
Y_test =one_hot_matrix(Y_test_orig).T
print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))
conv_layers = {}
