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
        with open(f'file\{mode}_set.txt') as f:
            for line in f.readlines():
                info = line.strip().split('\t')
                if len(info) > 0:
                    self.data.append([info[0].strip(), info[1].strip()])

    def __getitem__(self, index):

        image_file, lable = self.data[index]  #获取数据
        print(image_file)
        img = Image.open(image_file) # 读取图片、

        img = img.resize((224, 224), Image.ANTIALIAS) # 图片大小样式归一化统一调整像素为100*100
        plt.imshow(img)
        plt.show()
        img = np.array(img).astype('float32')  # 转换成数组类型浮点型32位
        # a = np.array([[3, 3, 3], [3, 4, 5]])
        # print(a.shape)  #输出为(2,3) 是一个两行三列的数组
        #print(img.shape)
        img = img.transpose((2, 0, 1)) # 读出来的图像是rgb， rgb， rgb 转置为 rrr...,ggg...,bbb...
        img = img/255.0 # 数据缩放到0-1的范围
        # 返回处理后的图片信息img 和分类类别 lable
        return img, np.array(lable)

    # python包含一个内置方法len（），使用它可以测量list、tuple等序列对象的长度
    def __len__(self):
        """
        获取样本
        """
        return len(self.data)

train_dataset = FoodDataset(mode='training')
for data, label in train_dataset:
    print(data)
    print(np.array(data).shape)  #输出矩阵的形状 是一个（3，100，100）的三维数组
    print(label)
    break