import os

#  分类将数据集中的猫和狗分离出来，用数据集划分
data = []
with open(f'file/train_set.txt') as f:
    for line in f.readlines():
        info = line.strip().split('\t')
        if len(info) > 0:
            data.append([info[0].strip(), info[1].strip()])
for path, cat_dag in data:
    if(cat_dag == "dog"):
        with open(f'file/dog_set.txt', 'a') as f_train:  # 新建文档以追加形式写入
            f_train.write(os.path.join(path) + '\t' + cat_dag + '\n')
    else:
        with open(f'file/cat_set.txt', 'a') as f_train:
            f_train.write(os.path.join(path) + '\t' + cat_dag + '\n')
