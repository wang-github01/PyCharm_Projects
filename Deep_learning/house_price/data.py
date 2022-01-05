import numpy as np
import json


#  数据集处理
def load_data():
    # 读入训练数据
    datafile = 'data/housing.data'
    # np.fromfile 用于读取函数 sep项目直接的分隔符
    data = np.fromfile(datafile, sep=' ')
    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                     'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

    feature_num = len(feature_names)

    # numpy函数中 shape[0] 读取矩阵中的行数
    # numpy函数中 shape[1] 读取矩阵中的列数
    # numpy函数中  reshape 是在不改变数据内容的情况下，改变一个数组的格式
    # 将原始数据进行reshape，变成[N，14]这样的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])
    #print(data.shape)
    # 将原始数据拆分成训练集和测试集
    # 这里使用80%的数据作为训练集、20%的数据作为测试基
    # 测试集和训练集必须没有交集（随机分配）
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    # data[:offset]相当于data[0:offset]取数据的前80%
    data_slice = data[:offset]

    # 计算train数据的最大值，最小值，平均值
    # 其中 axis=0表示每一列，axis=1表示每一行
    #maximums, minimums, avgs = data_slice.max(axis=0), data_slice.min(axis=0), data_slice.sum(axis=0) / data_slice.shape(axis=0)
    maximums = data_slice.max(axis=0)
    minimums = data_slice.min(axis=0)
    avgs = data_slice.sum(axis=0) / data_slice.shape[0]
    # print(data[:,1])
    # print("--------------")

    # 对数据进行归一化处理(做归一化处理的目的是 把数变为（0，1）之间的小数)
    for i in range(feature_num):
        # 对每一列进行归一化处理(归一化公式：(x - avgs)/(max - min))
        # data[2,2] 输出的是第三行，第三列的数据 。 data[:,2] 输出的是第三列的数据 。 data[2,:] 输出的是第三行的数据
        data[:,i] = (data[:,i] - avgs[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集划分
    # data[:offset] 是取前offset行数据，包括第offset行
    train_data = data[:offset]
    # data[offest:] 是取第offset行之后的所有行数据，不包括第offset行
    test_data = data[offset:]
    return train_data, test_data