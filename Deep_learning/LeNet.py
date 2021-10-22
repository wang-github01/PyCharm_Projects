# 图像分类
# 定义 LeNet 网络结构
import paddle
from paddle.nn import Conv2D, MaxPool2D, Linear
import paddle.nn.functional as F


class LeNet (paddle.nn.Layer):
    def __init__(self,num_classes=1):
        super(LeNet,self).__init__()
        # 创建卷积和池化层
        # 创建第一个卷积层
        # in_channels 定义输入通道数 ， out_channels 定义输出通道数。 kernel_size 卷积和大小。
        self.conv1 = Conv2D(in_channels=1, out_channels=6, kernel_size=5)
        # 创建最大池化层 2*2 ，kernel_size 最大池化窗口大小， seride 池化的步幅。
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        # 尺寸的逻辑：池化层未改变通道数：当前通道数为6
        # 创建第二个卷积层
        self.conv2 = Conv2D(in_channels=16, out_channels=16, kernel_size=5)
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
        # 创建第三个卷积层
        self.conv3 = Conv2D(in_channels=16, out_channels=120, kernel_size=4)
        # 尺寸逻辑：输入层将数据拉平[B,C,H,W] ->[B, C*H*W]
        # 输入size是[28, 28], 经过三次卷积和两次池化之后，C*H*W=120
        self.fc1 = Linear(in_features=120, out_features=64)
        # 创建全连接层，第一个全连接层的输出神经元个数为64， 第二个全连接层输出神经元个数为分类标签的类别数
        self.fc2 = Linear(in_features=64, out_features=num_classes)
    # 网络的先前计算过程
    def forward(self, x):
        x = self.conv1(x)
        # 每个卷积层使用Sigmoid激活函数，后面跟着一个2*2的池化
        x = F.sigmoid(x)
        x = self.max_pool1(x)
        x = F.sigmoid(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        # 尺寸逻辑：输入层将数据拉平[B, C, H, W] -> [B, C*H*W]
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        return x
