#序号
episodel = 0
while episodel < 20500:
    if episodel > 1024:
            if (episodel - 1024) % 200 == 0:
                with open("/kaggle/working/xh2w5.txt","a") as f:
                    dd=str(episodel)
                    f.write("{}\n".format(dd))
                    f.close()
    episodel +=1



#数据原本地!
file = open("/kaggle/working/ttr.txt")  #打开文档
data = file.readlines() #读取文档数据
listt = []  #新建列表，用于保存第一列数据
for num in data:
    listt.append(float(num))
# print(xh)
print(len(listt))

#所有用户的loss值在这里，那么下一步是分割出每一个的
#第一个用户loss
ue1 = listt[0:1967:8]
ue2 = listt[1:1967:8]
ue3 = listt[2:1967:8]
ue4 = listt[3:1967:8]
ue5 = listt[4:1967:8]
ue10 = listt[9:1967:8]
ue8 = listt[7:1967:8]
len(ue10)


#loss图 2e-5 2w5!
from pylab import *
# mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文

file = open('/kaggle/working/xh2w5.txt')#序号文件
dataa = file.readlines()
x = []
for num in dataa:
    x.append((float(num)))
# file = open('./s33.txt')
# data = file.readlines()
print(len(x))
# y1 = ue1
y2 = ue2

# for num in data:
#     y.append((float(num)))
# print(x)
# print(y)
# x_axis_data = [1, 2, 3, 4, 5]
# y_axis_data = [1, 2, 3, 4, 5]
xx = x[5:194]
yy = y2[5:194]
# print(len(x))
# print(len(y1))
# plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
plt.plot(xx, yy, linewidth=1)

# 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
plt.legend(loc="upper right")
plt.xlabel('Training  Steps')
plt.ylabel('Training  Loss')

plt.show()
# plt.savefig('demo.jpg')  # 保存该图片



