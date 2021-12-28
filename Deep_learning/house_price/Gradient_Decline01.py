import model
import data

#  梯度下降
train_data, test_data = data.load_data()
# x = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
x = train_data[:,:-1] # 将训练集前十三列赋值给 x
# y = ['MEDV']
y = train_data[:,-1:] # 将训练集的最后一列，也是结果复制给 y
net = model.Network(13)
# 此处可以一次性计算多个样本预测值和损失函数
x1 = x[0]
y1 = y[0]
z1 = net.forward(x1)
print('x1 {},shape {}'.format(x1, x1.shape))
print('y1 {},shape {}'.format(y1, y1.shape))
print('z1 {},shape {}'.format(z1, z1.shape))


# 计算每一个参数，w的梯度

gradient_w0 = (z1 - y1) * x1[0]

print((gradient_w0))
gradient_w = []
for i in range(13):
    gradient_w.append((z1 - y1) * x1[i])

print(gradient_w[0])