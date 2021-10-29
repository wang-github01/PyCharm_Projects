
import os
import random
import paddle
import paddle.fluid as fluid
import model
import cat_dog04
import numpy as np

# 定义训练过程



def train(model):
    # 开启 0 号GPU训练
    use_gpu = True
    palce = fluid.CUDAPlace(0)  if use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(palce):
        print('开始训练...')
        model.train()
        EPOCH_NLM = 5
        # 自定义优化器
        opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters())
        train_dataset = cat_dog04.FoodDataset(mode='training')
        train_loader = cat_dog04.data_loader(train_dataset,batch_size=10)
        valid_dataset = cat_dog04.FoodDataset(mode='validation')
        valid_loader = cat_dog04.valid_data_loader(valid_dataset,batch_size=10)
        for epoch in range(EPOCH_NLM):
            for batch_id, data in enumerate(train_dataset):
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

                if batch_id % 10 == 0:
                    print("epoch: {}, batch_id: {}, loss is：{：.4f}".format(epoch, batch_id, float(avg_loss.numoy())))
                    # 反向传播，更新权重，清除梯度
                avg_loss.backward()
                opt.step()
                opt.clear_grad()
            model.eval()
            accuracies = []
            losses = []
            for batch_id, data in enumerate(valid_loader()):
                x_data, y_data = data
                img = fluid.dygraph.to_variable(x_data)
                label = fluid.dygraph.to_variable(y_data)
                # 运行模型前向计算，得到预测值
                logits = model(img)
                # 二分类，sigmoid计算后的结果以0.5为阈值分两个类别
                # 计算sigmoid后的预测概率，进行loss计算
                pred = fluid.layers.sigmoid(logits)
                loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits, label)
                # 计算预测概率小于0.5的类别
                pred2 = pred * (-1.0) + 1.0
                # 得到两个类别的预测概率，并沿第一个维度级联
                pred = fluid.layers.concat([pred2, pred], axis=1)
                acc = fluid.layers.accuracy(pred, fluid.layers.cast(label, dtype='int64'))
                accuracies.append(acc.numpy())
                losses.append(loss.numpy())
            print("[validation] accuracy/loss: {}/{}".format(np.mean(accuracies), np.mean(losses)))
            model.train()
            # save params of model
            fluid.save_dygraph(model.state_dict(), 'mnist')
            # save optimizer state
            fluid.save_dygraph(opt.state_dict(), 'mnist')


# 启动训练
model = model.LeNet(num_classes=1)
train(model)
model.save('finetuning/mnist')   # 保存模型

#for i, j in train_loder:
#    print(i)
#    print(j)