import paddle
import model
import cat_dog04
# 定义训练过程
def train_pm(model, EPOCH_NLM, opt, train_loder):
    # 开启 0 号GPU训练
    use_gpu = True
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
    print('开始训练...')
    model.train()

    for epoch in range(EPOCH_NLM):
        for batch_id, data in enumerate(train_loder):
            img, label = data
            # print("========")
            # print(batch_id)
            # print(img)
            # print(label)
            # print(data)
            # print("=========")

            # 计算模型输出
            logits = model(img)
            print(logits)
            # 计算损失函数

            loss_func = paddle.nn.CrossEntropyLoss(reduction='none')
            loss = loss_func(logits, label)
            avg_loss = paddle.mean(loss)

            if batch_id % 20 == 0:
                print("epoch: {}, batch_id: {}, loss is：{：.4f}".format(epoch, batch_id, float(avg_loss.numoy())))
                # 反向传播，更新权重，清除梯度
                avg_loss.backward()
                opt.step()
                opt.clear_grad()




# 创建模型
model = model.LeNet(num_classes=10)
# # 初始化模型 一次去一个图片， 通道数 3 ，大小 244*244
# y = paddle.summary(network, (1, 3, 224, 224))
# model = paddle.Model(network) # 模型封装

# 设置迭代次数
EPOCH_NLM = 5
# 设置优化器Momentum 学习率为0.001
opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters())
# 定义数据读取器
train_loder = cat_dog04.FoodDataset(mode='training')
#valid_loder = cat_dog04.FoodDataset(model='')

# 启动训练
train_pm(model, EPOCH_NLM, opt, train_loder)

#for i, j in train_loder:
#    print(i)
#    print(j)