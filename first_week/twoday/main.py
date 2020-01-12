import matplotlib.pyplot as plt
import numpy as np

from DLStudy.first_week.twoday.lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

print(classes)
m_train = train_set_y.shape[1]  # 训练集图像的数量209张
m_test = test_set_y.shape[1]  # 测试集图像的数量50张
num_px = train_set_x_orig.shape[1]  # 图像表示为(209,64,64,3)，num_px表示图像的维数为64维

# print("训练集的数量: m_train = " + str(m_train))
# print("测试集的数量 : m_test = " + str(m_test))
# print("每张图片的宽/高 : num_px = " + str(num_px))
# print("每张图片的大小 : (" + str(num_px) + ", " + str(num_px) + ", 3)")
# print("训练集_图片的维数 : " + str(train_set_x_orig.shape))
# print("训练集_标签的维数 : " + str(train_set_y.shape))
# print("测试集_图片的维数: " + str(test_set_x_orig.shape))
# print("测试集_标签的维数: " + str(test_set_y.shape))

# 将训练集的维度降低并转置。变成2维，(12288,209)
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
# 将测试集的维度降低并转置。(12288,50)
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
# print("-----------------------------------------")
# print("训练集降维最后的维度： " + str(train_set_x_flatten.shape))
# print("训练集_标签的维数 : " + str(train_set_y.shape))
# print("测试集降维之后的维度: " + str(test_set_x_flatten.shape))
# print("测试集_标签的维数 : " + str(test_set_y.shape))

train_set_x = train_set_x_flatten / 255  # 255是最大的颜色通道值
test_set_x = test_set_x_flatten / 255


def sigmod(z):
    s = 1.0 / (1 + 1 / np.exp(z))
    return s


def initialize_with_zeros(dim):
    # w为权重，b为偏置
    w = np.zeros(shape=(dim, 1))
    b = 0
    # 使用断言来确保我要的数据是正确的
    assert (w.shape == (dim, 1))  # w的维度是(dim,1)
    assert (isinstance(b, float) or isinstance(b, int))  # b的类型是float或者是int
    return w, b


# 传播
def propagate(w, b, X, Y):
    # X为训练数据(64*64*3,209) Y为实际的标签矢量(1,209)
    # w为权重(64*64*3,1)，b为偏置
    # cost为损失成本值

    m = X.shape[1]  # 12288
    # 正向传播
    A = sigmod(np.dot(w.T, X) + b)  # 计算激活函数值。
    # 交叉熵损失函数
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # 计算成本

    # 反向传播
    dw = (1 / m) * np.dot(X, (A - Y).T)  # 请参考视频中的偏导公式。
    db = (1 / m) * np.sum(A - Y)  # 请参考视频中的偏导公式。

    # 使用断言确保我的数据是正确的
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    # 创建一个梯度字典，把dw和db保存起来。
    gradient = {
        "dw": dw,
        "db": db
    }
    return gradient, cost


# 优化
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []

    for i in range(num_iterations):

        gradient, cost = propagate(w, b, X, Y)

        dw = gradient["dw"]
        db = gradient["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db


        # 记录成本，每100个成本添加到成本数值中
        if i % 100 == 0:
            costs.append(cost)
        # 打印成本数据
        # if print_cost and (i % 100 == 0):
        #     print("迭代的次数: %i ， 误差值： %f" % (i, cost))

    params = {
        "w": w,
        "b": b}
    gradient = {
        "dw": dw,
        "db": db}
    return params, gradient, costs


def predict(w, b, X):
    # X是训练数据(64*64*64*3，209)
    m = X.shape[1]  # 图片的数量
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    # 预测猫在图片中出现的概率
    A = sigmod(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        # 将概率a [0，i]转换为实际预测p [0，i]
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
    # 使用断言
    assert (Y_prediction.shape == (1, m))

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):

    w, b = initialize_with_zeros(X_train.shape[0])

    parameters, gradient, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # 从字典“参数”中检索参数w和b
    w, b = parameters["w"], parameters["b"]

    # 预测测试/训练集的例子
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # 打印训练后的准确性
    print("训练集准确性：", format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100), "%")
    print("测试集准确性：", format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100), "%")

    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediciton_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations}
    return d



d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.006, print_cost=True)

costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

learning_rates = [0.005, 0.006, 0.007]
models = {}
for i in learning_rates:
    print("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=i,
                           print_cost=False)
    print('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
