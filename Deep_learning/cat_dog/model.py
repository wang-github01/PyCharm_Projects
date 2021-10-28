import paddle
import paddle.nn.functional as F


# 使用 LeNet神经网络
# 继承paddle.nn.Layer类，用于搭建模型
class LeNet(paddle.nn.Layer):
    def __init__(self, num_classes = 1):
        super(LeNet, self).__init__()

        # 创建卷积和池化层块， 每个卷积层使用Sigmoid激活函数， 后面跟着一个2*2的池化
        # in_channels 输入数据的通道数， out_channels 输出数据的通道数， kernel_size 卷积和大小， padding 零填充
        self.conv1 = paddle.nn.Conv2D(in_channels=3, out_channels=6, kernel_size=5)# 二维卷积层
        # kernel_size max pooling的窗口大小， stride 移动步长
        self.pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)  # 最大池化

        self.conv2 = paddle.nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)

        # 创建第 3 个卷积层
        self.conv3 = paddle.nn.Conv2D(in_channels=16, out_channels=120, kernel_size=4)

        # in_features 每个输入 x 的样本特征的大小 ，out_features  每个输出 y 的样本特征大小
        # 创建全连接层， 第一个全连接层的输出神经元为64
        self.fc1 = paddle.nn.Linear(in_features=300000, out_features=64)
        # 第二个全连接层输出神经元个数为分类标签的类别说
        self.fc2 = paddle.nn.Linear(in_features=64, out_features=num_classes)


# 网络前向计算
    def forward(self, x, label=None):
        # print(input.shape)
        # 将输入数据的样子该变成[1,3,100,100]
        x = paddle.reshape(input, shape=[1, 3, 224, 224])
        x = self.conv1(input) # 数据输入卷积层
        x = F.sigmoid(x)  #激活层
        x = self.pool1(x)  #池化层
        x = F.sigmoid(x)

        x = self.conv2(x)
        x = self.pool2(x)
        # print(x.shape)

        x = self.conv3(x)
        x = F.sigmoid(x)
        x = paddle.reshape(x, [x.shape[0], -1])

        x = self.fc1(x)  # 线性层
        x = F.sigmoid(x)
        x = self.fc2(x)
        return x