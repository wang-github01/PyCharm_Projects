import os
import random
import cv2
import paddle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def transform_file(datadir):
    data = []
    file = open(f'file\{datadir}_set.txt', 'r')
    FileNameList = file.readlines()  # file.readlines() 用于读取所有行，直到结束返回列表
    random.shuffle(FileNameList)  # 打乱顺序随机排序
    for line in FileNameList:  # 遍历列表
        info = line.strip().split('\t')  # strip()表示删除掉数据中的换行符，split（‘\t’）则是数据中遇到‘\t’ 就隔开。
        if len(info) > 0:
            data.append([info[0].strip(), info[1].strip()])  # 将列表中的元素，追加到date元组当中
    file.close()
    return data


def transform_img(img):
    #print(image_file)
    img = Image.open(img)  # 读取图片、
    # plt.imshow(img)
    # plt.show()
    img = img.resize((100, 100), Image.ANTIALIAS)  # 图片大小样式归一化统一调整像素为224*224
    img = np.transpose(img, (2, 0, 1))
    img = img.astype('float32')
    #img = np.array(img).astype('float32')  # 转换成数组类型浮点型32位
    # a = np.array([[3, 3, 3], [3, 4, 5]])
    # print(a.shape)  #输出为(2,3) 是一个两行三列的数组
    # print(img.shape)
    #img = img.transpose((2, 0, 1))  # 读出来的图像是rgb， rgb， rgb 转置为 rrr...,ggg...,bbb...
    img = img / 255.0  # 数据缩放到0-1的范围
    # 返回处理后的图片信息img 和分类类别 lable
    img = img * 2.0 - 1.0
    return img


def data_loader(datadir, batch_size):
    print("训练集数据处理。。。。")
    data_file = transform_file(datadir)
    #print(data_file)

    print("=============")
    batch_imgs = []
    batch_labels = []
    imgs_array1 = []
    labels_array1 = []
    for img, labels in data_file:
        if labels == "cat":
            label = 0
        elif labels == 'dog':
            label = 1
        #print(img)
        #print(labels)
        #print(label)
        imgs = transform_img(img)
        batch_imgs.append(imgs)
        batch_labels.append(label)
        # print(batch_labels)
        if len(batch_imgs) == batch_size:
            # 十条数据为一个数组
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('float32').reshape(-1,1)
            imgs_array1.append(imgs_array)
            labels_array1.append(labels_array)
            #yield  imgs_array, labels_array
            batch_imgs = []
            batch_labels = []
    if len(batch_imgs) > 0:
        imgs_array = np.array(batch_imgs).astype('float32')
        labels_array = np.array(batch_labels).astype('float32').reshape(-1,1)
        imgs_array1.append(imgs_array)
        labels_array1.append(labels_array)
        #yield  imgs_array, labels_array
    return imgs_array1,labels_array1

# 定义验证集读取器
def valid_data_loader(datadir, batch_size = 10):
    print("验证集数据处理。。。。。")
    data_file = transform_file(datadir)
    #print(data_file)

    print("=============")
    batch_imgs = []
    batch_labels = []
    imgs_array1 = []
    labels_array1 = []
    for img, labels in data_file:
        if labels == "cat":
            label = 0
        elif labels == 'dog':
            label = 1
        #print(img)
        #print(labels)
        #print(label)
        imgs = transform_img(img)
        batch_imgs.append(imgs)
        batch_labels.append(label)
        # print(batch_labels)
        if len(batch_imgs) == batch_size:
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('float32').reshape(-1,1)
            imgs_array1.append(imgs_array)
            labels_array1.append(labels_array)
            #yield  imgs_array, labels_array
            batch_imgs = []
            batch_labels = []
    if len(batch_imgs) > 0:
        imgs_array = np.array(batch_imgs).astype('float32')
        labels_array = np.array(batch_labels).astype('float32').reshape(-1,1)
        imgs_array1.append(imgs_array)
        labels_array1.append(labels_array)
        #yield  imgs_array, labels_array
    return imgs_array1,labels_array1

