import numpy as np
import model02
import data
import matplotlib.pyplot as plt

# 随机梯度下降

"""
# 使用 np.random.shuffle 打乱1维数组 和 2 维数组的元素顺序
a = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
print('before shuffle',a)
np.random.shuffle(a)
print('after shuffle',a)

b = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
b = b.reshape([6,2])
print('before shuffle\n',b)
np.random.shuffle(b)
print('after shuffle\n',b)
"""
"""
# 获取数据
train_data, test_data = data.load_data()

# 打乱样本顺序
np.random.shuffle(train_data)

# 将train_data分成多个mini_batch
batch_size = 10
n = len(train_data)
mini_batches = [train_data[k:k + batch_size] for k in range(0,n,batch_size)]

# 创建网络
net = model02.Network(13)

for mini_batch in mini_batches:
    x = mini_batch[:, :-1]  # 将训练集前十三列赋值给 x
    y = mini_batch[:, -1:]  # 将训练集的最后一列，也是结果复制给
    loss = net.train(x,y,itsdangerous=1)
# 设置训练次数
num_iterations=2000

# 启动训练
losses = net.train(x, y, iterations=num_iterations, eta=0.01)

# 画出损失函数的变化趋势
plot_x = np.arange(num_iterations)
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()
"""
# 获取数据
train_data, test_data = data.load_data()

# 创建网络
net = model02.Network(13)
# 启动训练
losses = net.train(train_data, num_epochs=100, batch_size=100, eta=0.1)

# 画出损失函数的变化趋势
plot_x = np.arange(len(losses))
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()