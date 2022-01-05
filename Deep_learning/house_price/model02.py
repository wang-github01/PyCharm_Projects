import numpy as np
class Network(object):
    def __init__(self, num_of_weights):

        #np.random.seed(0) # np.random.seed(n)函数用于生成指定随机数。
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
        N = x.shape[0]
        gradient_w = 1. / N * np.sum((z - y) * x,axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = 1. / N * np.sum((z - y))
        return gradient_w, gradient_b

    # 修改 参数 w 和 b
    def update(self, gardient_w, gardient_b, eta=0.01):
        self.w = self.w - eta * gardient_w
        self.b = self.b - eta * gardient_b

    def train(self, training_data, num_epochs, batch_size=10, eta=0.01):
        n = len(training_data)
        losses = []
        for epoch_id in range(num_epochs):
            # 在每轮迭代开始之前，将训练数据的顺序随机打乱
            # 然后再按每次取batch_size条数据的方式取出
            np.random.shuffle(training_data)
            # 将训练数据进行拆分，每个mini_batch包含batch_size条的数据
            mini_batches = [training_data[k:k + batch_size] for k in range(0, n, batch_size)]
            for iter_id, mini_batch in enumerate(mini_batches):
                # print(self.w.shape)
                # print(self.b)
                x = mini_batch[:, :-1]
                y = mini_batch[:, -1:]
                a = self.forward(x)
                loss = self.loss(a, y)
                gradient_w, gradient_b = self.gradient(x, y)
                self.update(gradient_w, gradient_b, eta)
                losses.append(loss)
                print('Epoch {:3d} / iter {:3d}, loss = {:.4f}'.
                      format(epoch_id, iter_id, loss))

        return losses