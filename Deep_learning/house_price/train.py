import numpy as np
import model
import data
import matplotlib.pyplot as plt

# 获取数据
train_data, test_data = data.load_data()
x = train_data[:, :-1]  # 将训练集前十三列赋值给 x
y = train_data[:, -1:]  # 将训练集的最后一列，也是结果复制给
# 创建网络
net = model.Network(13)
# 设置训练次数
num_iterations=2000

# 启动训练
losses = net.train(x, y, iterations=num_iterations, eta=0.01)

# 画出损失函数的变化趋势
plot_x = np.arange(num_iterations)
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()

