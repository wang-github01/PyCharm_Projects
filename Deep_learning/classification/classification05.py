import numpy as np
import paddle
from PIL import Image
import matplotlib.pyplot as plt

import classification04
import model

site = 255 # 读取图片位置
model_state_dict = paddle.load('finetuning/mnist.pdparams') # 读取模型

model1 =  model.MyCNN()# 实例化模型
model1.set_state_dict(model_state_dict)
model1.eval()

ceshi = model1(paddle.to_tensor(classification04.img))
print('预测结果为：', np.argmax(ceshi.numpy())) # 获取值
Image.open(classification04.img_path[site])  # 显示图片
