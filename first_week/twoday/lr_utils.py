import numpy as np
import h5py



def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features训练集中的图像数据
    train_set_y = np.array(train_dataset["train_set_y"][:])  # your train set labels训练集中的图像的队友分类标签

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features测试集中的图像数据
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels测试集中的图像的队友分类标签

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes保存的是以bytes类型保存的两个字符串数据，
    # 数据为：[b’non-cat’ b’cat’]

    train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))
    test_set_y = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes

# train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()
# index = 25
# plt.imshow(train_set_x_orig[index])
# plt.show()
