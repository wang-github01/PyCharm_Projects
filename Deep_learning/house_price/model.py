import numpy as np
import data
"""
    模型使用线性回归问题
        一层神经网络：z = W.x + b
        激活函数：y = activation_fn(z)

                                     """
# 这是一个没有预测能力的模型
# 模型设计--前向计算
class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性。
        # 此处设置固定的随机数种子
        np.random.seed(0) # np.random.seed(n)函数用于生成指定随机数。
        self.w = np.random.randn(num_of_weights, 1) #返回num_of_weights行，1列个符合正态分布的随机数
        #print(self.w)'
        #self.w[5] = -100.
        #self.w[9] = -100.
        self.b = 0.

    # forward 函数其实就是模型的前向计算过程
    def forward(self, x):
        # np.dot() 矩阵乘法
        z = np.dot(x, self.w) + self.b
        return z

    def loss(self, z, y):
        error = z - y
        num_samples = error.shape[0]
        #print(num_samples)
        cost = error * error
        cost = np.sum(cost) / num_samples
        return cost

    def gradient(self, x, y):
        z = self.forward(x)
        gradient_w = (z - y) * x
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (z - y)
        gradient_b = np.mean(gradient_b)
        return gradient_w, gradient_b

    def update(self, gardient_w, gardient_b, eta=0.01):
        self.w = self.w - eta * gardient_w
        self.b = self.b - eta * gardient_b

    def train(self,x, y, iterations=100,eta=0.01):
        points = []
        losses = []
        for i in range(iterations):
            points.append([self.w[5][0],self.w[9][0]])
            z = self.forward(x)
            L = self.loss(z, y)
            gradient_w, gradient_b = self.gradient(x, y)
            self.update(gradient_w, gradient_b, eta)
            losses.append(L)
            if i % 10 == 0:
                print('iter {}, loss {}'.format(i, L))
        return losses

'''
train_data, test_data = data.load_data()
# x = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
x = train_data[:,:-1] # 将训练集前十三列赋值给 x
# y = ['MEDV']
y = train_data[:,-1:] # 将训练集的最后一列，也是结果复制给 y
net = Network(13)
"""
一次计算一个样本
# x[0] 是输出训练集的前十三列的第一行数据，y[0]输出的是最后一列的第一行数据
# x1 = x[0]
# y1 = y[0]
"""
# 此处可以一次性计算多个样本预测值和损失函数
x1 = x[0:3]
y1 = y[0:3]
z = net.forward(x1)
print(z)

# 线性回归问题通常采用均方误差作为，评价模型好坏的指标 Loss = （y-z）^2 其中Loss通常也被称作损失函数，它是衡量模型好坏的指标
# loss = (y1 - z) * (y1 - z)
loss = net.loss(z, y1)
print('loss:',loss)
'''