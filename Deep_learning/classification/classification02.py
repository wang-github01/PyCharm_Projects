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
    # self 是指调用自己本身
    def __init__(self, mode):
        """print()
            初始化函数
        """
        self.data = []
        with open(f'{mode}_set.txt') as f:
            for line in f.readlines():
                info = line.strip().split('\t')
                if len(info) > 0:
                    self.data.append([info[0].strip(), info[1].strip()])


    def __getitem__(self, index):
        """
        读取图片，对图片进行归一化处理，返回图片和 标签
        """
        # 将图片路径给image_file , 分类名字个 lable
        print(index)
        image_file, lable = self.data[index]  #获取数据
        print(image_file)
        img = Image.open(image_file) # 读取图片
        img = img.resize((100, 100), Image.ANTIALIAS) # 图片大小样式归一化统一调整像素为100*100
        img = np.array(img).astype('float32')  # 转换成数组类型浮点型32位
        img = img.transpose((2, 0, 1)) # 读出来的图像是rgb， rgb， rgb 转置为 rrr...,ggg...,bbb...
        print
        img = img/255.0 # 数据缩放到0-1的范围
        # 返回处理后的图片信息img 和分类类别 lable
        return img, np.array(lable, dtype='int64')

    def __len__(self):
        """
        获取样本
        """

        return len(self.data)



if __name__ == '__main__':
    # 训练的数据提供器
    train_dataset = FoodDataset(mode='./classification/training')
    # 测试的数据提供器
    eval_dataset = FoodDataset(mode='./classification/validation')

    # 查看训练和测试数据的大小
    print('train大小：', train_dataset.__len__())
    print('eval大小：', eval_dataset.__len__())

    # 查看图片数据、大小及标签
    for data, label in train_dataset:
    #     print(data)
    #     print(np.array(data).shape)  #输出矩阵的形状
         print(label)
         break
    