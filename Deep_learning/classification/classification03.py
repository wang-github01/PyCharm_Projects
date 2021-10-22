import paddle
import os
import paddle.vision.transforms as T
import numpy as np
from PIL import Image
import paddle.nn.functional as F
import classification02

# 继承paddle.nn.Layer类，用于搭建模型
class MyCNN(paddle.nn.Layer):
    def __init__(self):
        super(MyCNN, self).__init__()
        # 二维卷积层
        self.conv0 = paddle.nn.Conv2D(in_channels=3, out_channels=20, kernel_size=5, padding=0)
        self.pool0 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)  # 最大池化
        self._batch_norm_0 = paddle.nn.BatchNorm2D(num_features=20) # 归一层

        self.conv1 = paddle.nn.Conv2D(in_channels=20, out_channels=50, kernel_size=5, padding=0)
        self.pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self._batch_norm_1 = paddle.nn.BatchNorm2D(num_features=50)

        self.conv2 = paddle.nn.Conv2D(in_channels=50, out_channels=50, kernel_size=5, padding=0)
        self.pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)

        self.fc1 = paddle.nn.Linear(in_features=4050, out_features=218) #  线形层
        self.fc2 = paddle.nn.Linear(in_features=218, out_features=100)
        self.fc3 = paddle.nn.Linear(in_features=100, out_features=11)

    def forward(self, input):
        # 将输入的数据变成该样子[1, 3, 100, 100]
        input = paddle.reshape(input,shape=[-1, 3, 100, 100]) #转换维度
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
        y =F.softmax(x) # 分类器
        return y


# if __name__ == '__main__':
#     network = MyCNN() # 模型实例化
#     print(paddle.summary(network, (1, 3, 100, 100)))

network = MyCNN()
y = paddle.summary(network, (1, 3, 100, 100))
print(y)
model = paddle.Model(network) # 模型封装

# 配置优化器、损失函数、评估指标
model.prepare(paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters()),
              paddle.nn.CrossEntropyLoss(),
              paddle.metric.Accuracy())

# 训练可视化VisualDL 工具的回调函数
visualdl= paddle.callbacks.VisualDL(log_dir='visualdl_log')

# 训练的数据提供器
train_dataset = classification02.FoodDataset(mode='training')
# 测试的数据提供器
eval_dataset = classification02.FoodDataset(mode='validation')

# 启动模型全流程训练
model.fit(train_dataset,                    # 训练数据集
          eval_dataset,                     # 评估数据集
          epochs=5,                         # 训练总次数
          batch_size=64,                    # 训练使用的批次大小
          verbose=1,                        # 日志形式展示
          callbacks=[visualdl])             # 设置可视化