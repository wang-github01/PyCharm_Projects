import model
import data
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

train_data, test_data = data.load_data()
# x = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
x = train_data[:,:-1] # 将训练集前十三列赋值给 x
# y = ['MEDV']
y = train_data[:,-1] # 将训练集的最后一列，也是结果复制给 y
net = model.Network(13)
# 此处可以一次性计算多个样本预测值和损失函数
x1 = x[0:3]
y1 = y[0:3]
losses = []

# 使用梯度下降方法进行模型训练
# 设置w5 和 w9的取值范围
w5 = np.arange(-160.0,160.0,1.0)
w9 = np.arange(-160.0,160.0,1.0)
losses = np.zeros([len(w5),len(w9)])
for i in range(len(w5)):
    for j in range(len(w9)):
        net.w[5] = w5[i]
        net.w[9] = w9[j]
        z = net.forward(x)
        loss = net.loss(z,y)
        losses[i,j] = loss

fig = plt.figure()
ax = Axes3D(fig)
w5, w9 = np.meshgrid(w5, w9)
ax.plot_surface(w5, w9, losses, rstride=1, cstride=1, cmap='rainbow')
plt.show()