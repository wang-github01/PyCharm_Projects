import numpy as np

import model
import data

#  梯度下降
train_data, test_data = data.load_data()
# x = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
x = train_data[:,:-1] # 将训练集前十三列赋值给 x
# y = ['MEDV']
y = train_data[:,-1:] # 将训练集的最后一列，也是结果复制给 y
net = model.Network(13)
# 注意这里是一次取出3个样本的数据，不是取出3个样本
x3samples = x[0:3]
y3samples = y[0:3]
z3samples = net.forward(x3samples)

print('----------------01-------------------')
gradient_w01 = (z3samples - y3samples) * x3samples
print('gradient_w {}, gradient.shape {}'.format(gradient_w01, gradient_w01.shape))

# 上面gradient_w02 的每一行代表了一个样本对梯度的贡献。根据梯度的计算公式，总梯度是对每个样本对梯度贡献的平均值。
print('----------------02-------------------')
z02 = net.forward(x)
gradient_w02 = (z02 - y) * x
print('gradient.shape {}'.format(gradient_w02.shape))


print('----------------03-------------------')
gradient_w03 = np.mean(gradient_w02, axis=0)
print('gradient_w02', gradient_w03.shape)
print('w', net.w.shape)
print(gradient_w03)
print(net.w)

print('----------------04-------------------')
# 将梯度维度转变成与参数维度y一致，方便加和运算
z03 = net.forward(x)
gradient_w03 = (z03 -y) * x
gradient_w03 = np.mean(gradient_w03, axis=0)
gradient_w03 = gradient_w03[:,np.newaxis]
print(gradient_w03.shape)

gradient_b = (z03 -y)
gradient_b = np.mean(gradient_b)
# 此处b是一个值，所以可以直接用np.mean得到一个标量
print(gradient_b)





