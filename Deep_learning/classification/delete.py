import os
# 新建标签列表
if(os.path.exists('./training_set.txt')):  # 判断有误文件
    os.remove('./training_set.txt')  # 删除文件
if(os.path.exists('./validation_set.txt')):
    os.remove('./validation_set.txt')
if(os.path.exists('./testing_set.txt')):
    os.remove('./testing_set.txt')