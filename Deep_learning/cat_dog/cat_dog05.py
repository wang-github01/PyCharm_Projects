import random

import numpy as np
from PIL import Image

def openimg():  # 读取图片
    file = open(f'file/testing_set.txt', 'r')
    FileNameList = file.readlines()  # file.readlines() 用于读取所有行，直到结束返回列表
    test_img =[]
    txt = []
    for line in FileNameList: # 循环读取每一行
        line = line.strip("\n")
        print(line)
        img = Image.open(line)  # 打开图片
        img = img.resize((100, 100), Image.ANTIALIAS)  # 图片大小样式归一化统一调整像素为224*224
        img = np.transpose(img, (2, 0, 1))
        img = img.astype('float32')
        img = img/255.0 # 缩放
        img = img * 2.0 - 1.0
        txt.append(line) # 生成列表
        test_img.append(img)
    return txt,test_img
img_path,img = openimg() # 读取列表
# print(img_path)