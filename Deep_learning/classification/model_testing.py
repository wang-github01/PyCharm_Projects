import random

import numpy as np
import paddle
from PIL import Image
import matplotlib.pyplot as plt

import classification03
import model

# site = 255 # 读取图片位置
model_state_dict = paddle.load('finetuning/mnist.pdparams') # 读取模型

model1 =  model.MyCNN()# 实例化模型
model1.set_state_dict(model_state_dict)
model1.eval()
index = random.sample(range(0,3000),50)
for site in index:
    ceshi = model1(paddle.to_tensor(classification03.img[site]))

    print('预测结果为：', np.argmax(ceshi.numpy())) # 获取值
# Image.open(classification03.img_path[site])  # 显示图片
    img = Image.open(classification03.img_path[site])
    #print(classification03.img_path[site])
    plt.imshow(img)
    plt.show()