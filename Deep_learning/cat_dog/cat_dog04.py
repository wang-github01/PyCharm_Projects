import os
import random

import paddle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# 继承paddle.io.Dataset对数据集做处理
class FoodDataset(paddle.io.Dataset):

    # self 是指调用自己本身
    #  __init__（）方法负责初始化，它是在对象创建时被解释器自动调用的
    def __init__(self, mode):
        """print()
            初始化函数
        """
        # 定义一个data 元组
        self.data = []
        file = open(f'file\{mode}_set.txt', 'r')
        FileNameList = file.readlines()  # file.readlines() 用于读取所有行，直到结束返回列表
        random.shuffle(FileNameList)  # 打乱顺序随机排序
        for line in FileNameList:  # 遍历列表
            info = line.strip().split('\t')  # strip()表示删除掉数据中的换行符，split（‘\t’）则是数据中遇到‘\t’ 就隔开。
            if len(info) > 0:
                self.data.append([info[0].strip(), info[1].strip()])  # 将列表中的元素，追加到date元组当中
        file.close()
    def __getitem__(self, index):

        image_file, lable = self.data[index]  # 获取数据
        #print(image_file)
        img = Image.open(image_file)  # 读取图片、
        # plt.imshow(img)
        # plt.show()
        img = img.resize((224, 224), Image.ANTIALIAS)  # 图片大小样式归一化统一调整像素为100*100
        img = np.array(img).astype('float32')  # 转换成数组类型浮点型32位
        # a = np.array([[3, 3, 3], [3, 4, 5]])
        # print(a.shape)  #输出为(2,3) 是一个两行三列的数组
        # print(img.shape)
        img = img.transpose((2, 0, 1))  # 读出来的图像是rgb， rgb， rgb 转置为 rrr...,ggg...,bbb...
        img = img / 255.0  # 数据缩放到0-1的范围
        # 返回处理后的图片信息img 和分类类别 lable
        return img, np.array(lable)

    # python包含一个内置方法len（），使用它可以测量list、tuple等序列对象的长度
    def __len__(self):
        """
        获取样本
        """
        return len(self.data)


def data_loader( data, batch_size=10):
    def reader():
        batch_imgs = []
        batch_labels = []
        for img, labels in data:
            if labels == "cat":
                label = 0
            elif labels == 'dog':
                label = 1
            batch_imgs.append(img)
            batch_labels.append(label)
        # print(batch_labels)
            if len(batch_imgs) == batch_size:
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('float32').reshape(-1,1)
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []
            if len(batch_imgs) > 0:
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('float32').reshape(-1,1)
                yield  imgs_array, labels_array
    return reader

# 定义验证集读取器
def valid_data_loader(data, batch_size = 10,):
    def reader():
        batch_imgs = []
        batch_labels = []
        for img, labels in data:
            if labels == "cat":
                label = 0
            elif labels == 'dog':
                label = 1
            batch_imgs.append(img)
            batch_labels.append(label)
        # print(batch_labels)
            if len(batch_imgs) == batch_size:
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('float32').reshape(-1,1)
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []
            if len(batch_imgs) > 0:
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('float32').reshape(-1,1)
                yield  imgs_array, labels_array
    return reader
    # def reader():
    #     if mode == 'train':
    #         random.shuffle(filenames)
    #     batch_imgs = []
    #     batch_labels = []
    #     for img, labels in data:
    #         print(img)
    #         print(labels)


#train_dataset = FoodDataset(mode='training')
# print(train_dataset)
#daradir = 'img'
#data_loader(daradir, train_dataset, batch_size=0, mode='train')
# for data, label in train_dataset:
#     print(data)
#     print(np.array(data).shape)  #输出矩阵的形状 是一个（3，100，100）的三维数组
#     print(label)
#     break
