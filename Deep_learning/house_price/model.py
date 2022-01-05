import numpy as np
import data
"""
    模型使用线性回归问题
        一层神经网络：z = W.x + b
        激活函数：y = activation_fn(z)

    全流程
        1、通过前向计算拿到预测输出z（模型的预测值），根据模型的预测值跟模型的样本的真实的y值。
        2、计算loss（z = w * x + b, erroy = z - y, cost = error * error,loss = np.mean(cost)）。
        3、根据预测值和loss去计算梯度gradient。
        4、根据梯度更新我们的参数。
"""
# 这是一个没有预测能力的模型
# 模型设计--前向计算
class Network(object):
    def __init__(self, num_of_weights):

        np.random.seed(0) # np.random.seed(n)函数用于生成指定随机数。
        self.w = np.random.randn(num_of_weights, 1) #返回num_of_weights行，1列个符合正态分布的随机数
        self.b = 0.

    # forward 函数其实就是模型的前向计算过程
    def forward(self, x):
        # np.dot() 矩阵乘法
        # z = w * x + b
        z = np.dot(x, self.w) + self.b
        return z

    # 计算损失函数(使用均方差函数)
    def loss(self, z, y):
        error = z - y
        num_samples = error.shape[0]
        #print(num_samples)
        cost = error * error
        cost = np.sum(cost) / num_samples
        return cost

    # 梯度下降
    def gradient(self, x, y):
        z = self.forward(x)
        gradient_w = (z - y) * x
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (z - y)
        gradient_b = np.mean(gradient_b)
        return gradient_w, gradient_b

    # 修改 参数 w 和 b
    def update(self, gardient_w, gardient_b, eta=0.01):
        self.w = self.w - eta * gardient_w
        self.b = self.b - eta * gardient_b

    # 训练
    def train(self,x, y, iterations=100,eta=0.01):
        losses = []
        for i in range(iterations):
            z = self.forward(x)
            L = self.loss(z, y)
            gradient_w, gradient_b = self.gradient(x, y)
            self.update(gradient_w, gradient_b, eta)
            losses.append(L)
            if (i+1) % 10 == 0:
                print('iter {}, loss {}'.format(i, L))
        return losses