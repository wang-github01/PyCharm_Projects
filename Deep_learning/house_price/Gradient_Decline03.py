import numpy as np

import model
import data


#  梯度下降
train_data, test_data = data.load_data()
# x = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
x = train_data[:,:-1] # 将训练集前十三列赋值给 x
# y = ['MEDV']
y = train_data[:,-1:] # 将训练集的最后一列，也是结果复制给 y
# 初始化网络
net = model.Network(13)
# 设置[w5, w9] = [-100., -100.]
net.w[5] = -100.0
net.w[9] = -100.0


"""
    全流程
        1、通过前向计算拿到预测输出z（模型的预测值），根据模型的预测值跟模型的样本的真实的y值。
        2、计算loss（z = w * x + b, erroy = z - y, cost = error * error,loss = np.mean(cost)）。
        3、根据预测值和loss去计算梯度gradient。
        4、根据梯度更新我们的参数。
"""

# 1、拿到模型预测输出值 z
z = net.forward(x)

# 2、计算 loss
loss = net.loss(z, y)

# 3、
gradient_w, gradient_b = net.gradient(x, y)

gradient_w5 = gradient_w[5][0]
gradient_w9 = gradient_w[9][0]
print('point {} loss {}'.format([net.w[5][0], net.w[9][0]],loss))
print('gradient {}'.format([gradient_w5, gradient_w9]))

# 沿着梯度的反方向移动一小步下下一个点P1，观察损失函数的变化

# 在[w5, w9]平面上，沿着梯度的反方向移动到下一个点P1
# 定义移动步长eta

eta =0.1
# 更新参数w5和w9
for i in range(10000):
    print(i)
    net.w[5] = net.w[5] - eta * gradient_w5
    net.w[9] = net.w[9] - eta * gradient_w9

    # 重新计算 z 和 loss
    z = net.forward(x)
    loss = net.loss(z, y)

    gradient_w, gradient_b = net.gradient(x, y)

    gradient_w5 = gradient_w[5][0]
    gradient_w9 = gradient_w[9][0]
    print('point {} loss {}'.format([net.w[5][0], net.w[9][0]],loss))
    print('gradient {}'.format([gradient_w5, gradient_w9]))