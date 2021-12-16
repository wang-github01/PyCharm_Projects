import random

import numpy as np
import paddle
from PIL import Image
import matplotlib.pyplot as plt
import cat_dog05
import LeNet

# site = 255 # 读取图片位置
from cat_dog import LeNet_test

model_state_dict = paddle.load('finetuning/mnist.pdparams') # 读取模型

model1 = LeNet.LeNet(num_classes=1)# 实例化模型
model1.set_state_dict(model_state_dict)
model1.eval()
# 从1~3000 中每50 取一个数
# index = random.sample(range(1,3000),50)

for site in range(1,35):
    print(site)
    ceshi = model1(paddle.to_tensor(cat_dog05.img[site]))
    if (np.argmax(ceshi.numpy())==0):
        print('预测结果为：这是一只猫')
    else:
        print('预测结果为：这是一只狗')
    #print('预测结果为：', np.argmax(ceshi.numpy())) # 获取值
# Image.open(classification03.img_path[site])  # 显示图片
    img = Image.open(cat_dog05.img_path[site])
    #print(classification03.img_path[site])
    plt.imshow(img)
    plt.show()