import paddle
import paddle.fluid as fluid
import LeNet
import cat_dog04
import numpy as np
import sys
import matplotlib.pyplot as plt
from cat_dog import log
# 定义训练过程
import AlexNet


def train(model, EPOCH_NLM, opt):
    data_loss = []
    data_losses = []
    data_acc = []

    # 开启 0 号GPU训练
    use_gpu = True
    palce = fluid.CUDAPlace(0)  if use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard():
        print('开始训练...')
        model.train()
        #train_loader = cat_dog04.data_loader(datadir='training', batch_size=10)
        #valid_loader = cat_dog04.valid_data_loader(datadir='validation', batch_size=10)
        train_loader = cat_dog04.data_loader(datadir='test', batch_size=10)
        valid_loader = cat_dog04.valid_data_loader(datadir='test', batch_size=10)
        for epoch in range(EPOCH_NLM):

            print(epoch)
            data_x,data_y = train_loader
            batch_id = 0
            for imgs, labels in zip(data_x,data_y):
                img = fluid.dygraph.to_variable(imgs)
                label = fluid.dygraph.to_variable(labels)
                # 计算模型输出
                logits = model(img)

                # 计算损失函数
                loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits,label)
                avg_loss = fluid.layers.mean(loss)

                if batch_id % 10 == 0:
                    data_loss.append(float(avg_loss.numpy()))
                    print("epoch: {}, batch_id: {}, loss is：{:.4f}".format(epoch,batch_id,float(avg_loss.numpy())))
                batch_id = batch_id + 1
                    # 反向传播，更新权重，清除梯度
                avg_loss.backward()
                opt.minimize(avg_loss)
                model.clear_gradients()

            model.eval()
            accuracies = []
            losses = []
            data_x, data_y = valid_loader
            for imgs, labels in zip(data_x, data_y):
                img = fluid.dygraph.to_variable(imgs)
                label = fluid.dygraph.to_variable(labels)
                # 运行模型前向计算，得到预测值
                logits = model(img)
                # 二分类，sigmoid计算后的结果为0.5为阈值分两个类别
                # 计算sigmoid后的预测概率，进行loss计算
                pred = fluid.layers.sigmoid(logits)
                loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits, label)
                #pred = fluid.layers.relu(logits)
                pred2 = pred * (-0.1) + 1.0
                pred = fluid.layers.concat([pred2, pred], axis=1)
                # fluid.layers.cast(label, dtype='int64') 是将float32的label 转换成 int64
                acc = fluid.layers.accuracy(pred, fluid.layers.cast(label, dtype='int64'))
                # print(acc)
                accuracies.append(acc.numpy())
                losses.append(np.mean(loss.numpy()))
            acc_2 = np.mean(accuracies)
            losses_2 = np.mean(losses)

            data_losses.append(losses_2)
            data_acc.append(acc_2)
            print("[validation] accuracy {:.4f}, loss {:.4f}".format(float(acc_2), float(losses_2)))
            model.train()

        # 保存模型
        # save params of model
        fluid.save_dygraph(model.state_dict(), 'finetuning/mnist')
        # save optimizer state
        fluid.save_dygraph(opt.state_dict(), 'finetuning/mnist')
    plt.plot(np.arange(len(data_loss)),data_loss,label ='loss')
    plt.savefig('result/loss.jpg')
    plt.show()
    plt.plot(np.arange(len(data_acc)), data_acc,color='r',label ='acc')
    plt.plot(np.arange(len(data_losses)), data_losses,color='g',label='losses')
    plt.legend(loc = 'best')
    plt.savefig('result/acc_losses.jpg')
    plt.show()
#model.save('finetuning/mnist')
if __name__ == '__main__':
    sys.stdout = log.Logger(sys.stdout)  # 将输出记录到log
    sys.stderr = log.Logger(sys.stderr)  # 将错误信息记录到log
    #model = AlexNet.AlexNet("AlexNet")
    model = LeNet.LeNet(num_classes=1)

    # 调用模型
    EPOCH_NLM = 5

    # 自定义优化器
    opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters())

    train(model, EPOCH_NLM, opt)
        # model.save('finetuning/mnist')