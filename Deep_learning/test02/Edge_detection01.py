# 边缘检测
import paddle
import numpy as np
from paddle.nn import Conv2D
from paddle.nn.initializer import Assign
from PIL import Image
import matplotlib.pyplot as plt
import paddle.fluid as fluid

# 用PIL方式读取图片
img = Image.open('img/car1.jpg')
# 设置卷积和参数
# 创建一个3*3的数组
w = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')/8
w = w.reshape([1, 1, 3, 3])
# 由于输入通道数是3，将卷积核的形状从[1,1,3,3]调整为[1,3,3,3]
w = np.repeat(w, 3, axis=1)
# 创建卷积算子，输出通道数为1，卷积核大小为3*3
# 并使用上面设置好的参数值作为卷积和权重初始化参数
conv = Conv2D(in_channels=3, out_channels=1, kernel_size=[3,3], weight_attr=paddle.ParamAttr(
    initializer=Assign(value=w)))
# 将读入的图片转换为float32类型的numpy.ndarray

x = np.array(img).astype('float32')
# 图片读成ndarry时，形状是[H, W, 3]
# 将通道这一维度调整到最前面
x = np.transpose(x, (2, 0, 1))
# 将数据的形状调整为[N, C, H, W]
x = x.reshape(1, 3, img.height, img.width)
x = fluid.dygraph.to_variable(x)

y = conv(x)
out = y.numpy()

plt.figure(figsize=(20, 10))
f = plt.subplot(121)
f.set_title('input_img', fontsize=15)
plt.imshow(img)

f = plt.subplot(122)
f.set_title('out_img', fontsize=15)
plt.imshow(out.squeeze(), cmap='gray')
plt.show()