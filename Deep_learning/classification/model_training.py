import paddle
import classification02
import model


# 训练模型



# 调用模型
network = model.MyCNN()
# paddle.summary 打印模型
y = paddle.summary(network, (1, 3, 100, 100))
print(y)
model = paddle.Model(network) # 模型封装
print(model)

# 配置优化器、损失函数、评估指标
model.prepare(paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters()),
              paddle.nn.CrossEntropyLoss(),
              paddle.metric.Accuracy())

# 训练可视化VisualDL 工具的回调函数
visualdl= paddle.callbacks.VisualDL(log_dir='visualdl_log')

# 训练的数据提供器
train_dataset = classification02.FoodDataset(mode='training')
# 测试的数据提供器
eval_dataset = classification02.FoodDataset(mode='validation')

# 启动模型全流程训练
model.fit(train_dataset,                    # 训练数据集
          eval_dataset,                     # 评估数据集
          epochs=5,                         # 训练总次数
          batch_size=64,                    # 训练使用的批次大小
          verbose=1,                        # 日志形式展示
          callbacks=[visualdl])             # 设置可视化
model.save('finetuning/mnist')   # 保存模型