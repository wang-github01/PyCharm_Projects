#encoding=utf-8
import pandas as pd
import openpyxl
import xlsxwriter
# 读取数据
data = pd.read_csv("waimai_10k.csv",encoding="utf-8")
# 输出数据的维度（11987行，2列）
print(data)
print(data.shape)
# 输出数据众的头部信息
print(data.columns)
# 查询数据将label = 0 的数据筛选出来，
data_0 = data[data["label"]==0]
print(data_0)
# 将查询的数据保存起来名字为data_0.csv
data_0.to_excel(r"data_0.xlsx",encoding="utf-8",engine='xlsxwriter')
data_1 = data[data["label"]==1]
print(data_1)
data_1.to_excel(r"data_1.xlsx",encoding="utf-8", engine='xlsxwriter')
#data_1 = data["label=1"]
# print(data["label"])