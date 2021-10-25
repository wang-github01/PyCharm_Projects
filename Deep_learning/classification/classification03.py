import numpy as np
from PIL import Image

def openimg():  # 读取图片
    with open(f'testing_set.txt') as f: #读取文件
        test_img =[]
        txt = []
        for line in f.readlines(): # 循环读取每一行
            img = Image.open(line[:-1])  # 打开图片
            img = img.resize((100, 100), Image.ANTIALIAS) # 大小归一化
            img = np.array(img).astype('float32') # 转成数组
            img = img.transpose((2, 0, 1))
            img = img/255.0 # 缩放
            txt.append(line[:-1]) # 生成列表
            test_img.append(img)
        return txt,test_img
img_path,img = openimg() # 读取列表
# print(img_path)