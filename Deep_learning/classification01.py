import paddle
import os
import paddle.vision.transforms as T
import numpy as np
from PIL import Image
import paddle.nn.functional as F

# 读取图片名称并生成文档

data_path = 'food-11'  # 设置初始文件地址
# os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
character_folders = os.listdir(data_path)  # 查看地址下载文件
# testing(测试集)，traintng(训练集),Validation(验证集)
# print(character_folders)
data = '10_alken'
data[0:data.rfind('_', 1)] #判断_位置截取下划线前的数据

# 新建标签
for character_folder in character_folders:  # 循环文件夹列表
    with open(f'{character_folder}_set.txt','a') as f_train: # 新建文档以追加形式写入
        # os.path.join() 函数用于路径拼接文件路径,可以传入多个参数
        character_imgs = os.listdir(os.path.join(data_path, character_folder)) # 读取文件夹
        count = 0
        if character_folder in 'testing': # 检查是否是训练集
            for img in character_imgs:  # 循环读取列表
                f_train.write(os.path.join(data_path, character_folder, img) + '\n') #吧地址写入文档
                count += 1
#                print(character_folder, count) # 输出文件夹及图片数量
        else:
            for img in character_imgs:
                f_train.write(os.path.join(data_path, character_folder, img) + '\t' + img[0:img.rfind('_', 1)] + '\n') #写入地址标签
                count += 1
#                print(character_folder,count)  # 输出文件夹及图片数量