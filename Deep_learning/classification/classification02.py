import paddle
import numpy as np
from PIL import Image

# 测验下面类中__init__ 输出内容
'''
with open(f'./classification/training_set.txt') as f:  # 查看文件内容
    for line in f.readlines(): # 逐行读取
        # strip()表示删除掉数据中的换行符，split（‘\t’）则是数据中遇到‘\t’ 水平制表符 就隔开
        info = line.strip().split('\t') # 以\t 为换行符生成列表
        print(info)
        if len(info) > 0:  # 列表不为空
            print([info[0].strip(), info[1].strip()])  # 输出内容
'''

# 继承paddle.io.Dataset对数据集做处理
class FoodDataset(paddle.io.Dataset):

    """
    _init__（）用于类的初始化，几乎在任何框架定义类时都避免不了使用它，因为它负责创建类的实例属性并进行赋值等重要操作，
    尽管在新建对象时并不需要“显式”调用这个函数。
     如果需要自定义Dataset，就需要实现__getitem__（）和__len__（））
    """
    # self 是指调用自己本身
    #  __init__（）方法负责初始化，它是在对象创建时被解释器自动调用的
    def __init__(self, mode):
        """print()
            初始化函数
        """
        # 定义一个data 元组
        self.data = []
        with open(f'{mode}_set.txt') as f:
            for line in f.readlines():
                info = line.strip().split('\t')
                if len(info) > 0:
                    self.data.append([info[0].strip(), info[1].strip()])

    # __getitem__()方法接收一个idx参数，也就是自己给的索引值
    '''
            __getitem__(self,key):
            如果在类中定义了__getitem__()方法，那么它的实例对象（假设为P）就可以以P[key]形式取值，
            当实例对象做P[key]运算时，就会调用类中的__getitem__()方法。当对类的属性进行下标的操作时，首先会被__getitem__() 拦截，
            从而执行在__getitem__()方法中设定的操作，如赋值，修改内容，删除内容等。
    '''
    def __getitem__(self, index):
        """
        读取图片，对图片进行归一化处理，返回图片和 标签
        """
        # 将图片路径给image_file , 分类名字个 lable
        # print(index)
        image_file, lable = self.data[index]  #获取数据
        print(image_file)
        img = Image.open(image_file) # 读取图片、
        #plt.imshow(img)
        #plt.show()
        img = img.resize((100, 100), Image.ANTIALIAS) # 图片大小样式归一化统一调整像素为100*100
        img = np.array(img).astype('float32')  # 转换成数组类型浮点型32位
        # a = np.array([[3, 3, 3], [3, 4, 5]])
        # print(a.shape)  #输出为(2,3) 是一个两行三列的数组
        #print(img.shape)
        img = img.transpose((2, 0, 1)) # 读出来的图像是rgb， rgb， rgb 转置为 rrr...,ggg...,bbb...
        img = img/255.0 # 数据缩放到0-1的范围
        # 返回处理后的图片信息img 和分类类别 lable
        return img, np.array(lable, dtype='int64')

    # python包含一个内置方法len（），使用它可以测量list、tuple等序列对象的长度
    def __len__(self):
        """
        获取样本
        """
        return len(self.data)




# #     # 训练的数据提供器
train_dataset = FoodDataset(mode='training')
# #
# eval_dataset = FoodDataset(mode='validation')
#      # 查看训练和测试数据的大小
# print('train大小：', train_dataset.__len__())
# print('eval大小：', eval_dataset.__len__())
# #     # 查看图片数据、大小及标签
for data, label in train_dataset:
      print(data)
      print(np.array(data).shape)  #输出矩阵的形状 是一个（3，100，100）的三维数组
      print(label)
      break
