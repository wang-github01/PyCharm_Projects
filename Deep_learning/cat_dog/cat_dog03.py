import random

import os
import shutil
import numpy as np
import pandas as pd



# 定义一个函数用于读取文件将文件中的，顺序打乱，保存到 date 元组当中
def ReadFileDatas(original_filename):
    data = []
    file = open(original_filename,'r')    # 以只读形式打开文件
    FileNameList=file.readlines()         # file.readlines() 用于读取所有行，直到结束返回列表
    random.shuffle(FileNameList)          # 打乱顺序随机排序
    for line in FileNameList:             # 遍历列表
        info = line.strip().split('\t')   # strip()表示删除掉数据中的换行符，split（‘\t’）则是数据中遇到‘\t’ 就隔开。
        if len(info) > 0:
            data.append([info[0].strip(), info[1].strip()])  # 将列表中的元素，追加到date元组当中
    file.close()                          # 关闭文件
    print("数据总量：", len(FileNameList))
    return data

# 定义一个函数用于划分数据集，处理好的数据传入
def TrainValTestFile(FileNameList):
    l_val = []
    l_train = []
    i = 0
    j = len(FileNameList)
    # 这里将数据集划分为训练集和验证集比例为 7:3
    for line in FileNameList:
        if i<(j*0.7):      # 划分训练集
            i += 1
            l_train.append(line)
        else:              # 划分验证集
            i += 1
            l_val.append(line)
    print("总数量：%d，此时创建train,val数据集"%i)
    return l_train,l_val     # 返回的是，划分好后的数据结果


# 定义一个函数将划分好后的数据，保存到txt文件当中  listInfo 为划分好后的数据，new_filename 为要保存到的文件
def WriteDatasToFile_move(listInfo, new_filename):
    file_handle = open(new_filename, 'a')   # 打开一个文件用于追加。
    for str_Result in listInfo:             # 遍历数据集
        # write() 方法用于向文件中写入指定字符串，os.path.join() 函数用于路径拼接文件路径,可以传入多个参数
        file_handle.write(os.path.join(str_Result[0] + "\t" + str_Result[1] + "\n"))
        # shutil.move() 将文件 移动到另一个文件夹中
        shutil.move(str_Result[0].strip("\n"), r"img\valid")
    file_handle.close()                   # 关闭打开的文件
    print('写入 %s 文件成功.' % new_filename)

# 定义一个函数将划分好后的数据，保存到txt文件当中  listInfo 为划分好后的数据，new_filename 为要保存到的文件
def WriteDatasToFile(listInfo, new_filename):
    file_handle = open(new_filename, 'a')   # 打开一个文件用于追加。
    for str_Result in listInfo:             # 遍历数据集
        # write() 方法用于向文件中写入指定字符串，os.path.join() 函数用于路径拼接文件路径,可以传入多个参数
        file_handle.write(os.path.join(str_Result[0] + "\t" + str_Result[1] + "\n"))
        # shutil.move() 将文件 移动到另一个文件夹中
    file_handle.close()                   # 关闭打开的文件
    print('写入 %s 文件成功.' % new_filename)


listFileCat = ReadFileDatas(r'file\cat_set.txt')            # 处理猫数据集
l_train_cat, l_val_cat= TrainValTestFile(listFileCat)     # 将猫数据集划分

listFileDog = ReadFileDatas(r'file\dog_set.txt')            # 处理狗数据集
l_train_dog, l_val_dog = TrainValTestFile(listFileDog)     # 将狗的数据集划分

# 将划分好的验证集，保存到vaidation_set.txt中，并将对于的图片保存到 img\valid 下

WriteDatasToFile_move(l_val_cat, r'file\validation_set.txt')
WriteDatasToFile_move(l_val_dog, r'file\validation_set.txt')

# 将划分好后的训练集， 保存到training_set.txt中

WriteDatasToFile(l_train_cat, r'file\training_set.txt')
WriteDatasToFile(l_train_dog, r'file\training_set.txt')