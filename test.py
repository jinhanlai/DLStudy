# _*_ coding:utf-8 _*_
"""
 @author: LaiJinHan
 @time：2019/8/6 12:08
"""
import numpy as np
import matplotlib.image as mpimg
from PIL import Image


def change_image_channels(image, image_path):
    # 4通道转3通道
    if image.mode == 'RGBA':
        r, g, b, a = image.split()
        image = Image.merge("RGB", (r, g, b))
        image.save(image_path)
    return image

# path='D:\\PythonProject\\DLStudy\second_week\\threeday\\images\\5.png'
# img = Image.open(path)
# new_image = change_image_channels(img, path)
#
# image1 = mpimg.imread('D:\\PythonProject\\DLStudy\second_week\\threeday\\images\\2.png')
# print(image1.shape)

a=[1,2,3]
print(a.shape)
