import paddle
import paddle.nn.functional as F


# 使用 MyCNN 神经网络
# 继承paddle.nn.Layer类，用于搭建模型
class MyCNN(paddle.nn.Layer):
    def __init__(self):
        super(MyCNN, self).__init__()
        # in_channels 输入数据的通道数， out_channels 输出数据的通道数， kernel_size 卷积和大小， padding 零填充
        self.conv0 = paddle.nn.Conv2D(in_channels=3, out_channels=20, kernel_size=5, padding=0)# 二维卷积层
        # kernel_size max pooling的窗口大小， stride 移动步长
        self.pool0 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)  # 最大池化
        # num_features 指明通道数量
        self._batch_norm_0 = paddle.nn.BatchNorm2D(num_features=20) # 归一层

        self.conv1 = paddle.nn.Conv2D(in_channels=20, out_channels=50, kernel_size=5, padding=0)
        self.pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self._batch_norm_1 = paddle.nn.BatchNorm2D(num_features=50)

        self.conv2 = paddle.nn.Conv2D(in_channels=50, out_channels=50, kernel_size=5, padding=0)
        self.pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        #in_features 每个输入 x 的样本特征的大小 ，out_features  每个输出 y 的样本特征大小
        self.fc1 = paddle.nn.Linear(in_features=4050, out_features=218) #  线形层
        self.fc2 = paddle.nn.Linear(in_features=218, out_features=100)
        self.fc3 = paddle.nn.Linear(in_features=100, out_features=11)

    def forward(self, input):
        # 将输入的数据变成该样子[1, 3, 100, 100]
        input = paddle.reshape(input,shape=[1, 3, 100, 100]) #转换维度
        # print(input.shape)
        x = self.conv0(input) # 数据输入卷积层
        x = F.relu(x)  #激活层
        x = self.pool0(x)  #池化层
        x = self._batch_norm_0(x) #归一层

        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self._batch_norm_1(x)
        # print(x.shape)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = paddle.reshape(x, [x.shape[0], -1])

        x = self.fc1(x)  # 线性层
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        y =F.softmax(x) # 分类器  一行一行的做皈依化
        return y