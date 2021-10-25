import os
#  修改图片名字
files = os.listdir("img")
i=0
for file in files:
    original = "img" + os.sep+ files[i]
    new = "img" + os.sep + "img" + str(i+1) + ".jpg"
    os.rename(original,new)
    i+=1