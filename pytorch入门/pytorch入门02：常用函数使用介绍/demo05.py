#各种常用函数
import numpy as np
import torch

#获取张量数据类型
a = torch.randn(2, 3) #返回-1到1之间的随机数2行3列
print(a)
b = a.shape  #返回a形状 touch.Size([2,3])
print(b)
c = a.size() #返回的值和shape是一样的
d = a.size(1) #放回size(【2,3】）中第二个元素 即3，a.shape(1)返回的值一样

a = torch.rand(2, 2, 3)  #创建一个3维张量，可以理解为有2层，每层有2行3列的张量
print(a)
print(a.shape)    #返回的是torch.Size([2,2,3])
print(a[0])     #相等于取了第一层的数据，就是返回了第一层2行3列的张量
print(list(a.shape))    #更改a.shape为列表
print(a.numel())       #放回所占空间大小 2*2*3=12

a = np.array([2, 3, 3])  #我们导入的数据往往是numpy类型
torch.from_numpy(a)    #对a进行装换为tensor类型

print(torch.linspace(0,10, steps=11))  #等分0到10输出，共11个数
#切片
a = torch.rand(5, 3, 28, 28)  #4维张量，以传入的照片为例，相当于5张照片,每张照片有3层通道，每层通道是28*28像素大小
print(a[0,0].shape)   #相当于放回了第一张图片的第一层通道数据 所以结果是torch.Size([28,28])
print(a[0,0,2,4])    #相当于输出了第一张图片的第一层通道，第3行第5列的像素值大小
print(a[:2,1:,-1:,:].shape)  #第一个相当于从0到2就是取前2张图片，第二个是从1开始到最后，因为这里总共就3个通道所以输出2，
# 第三个-1表示反向索引，则从最后一个开始往右，返回1，4是取全部，所以这里输出是28 最后输出结果为torch.Size[2,2,1,28]

#维度变换
a = torch.rand(4,1,28,28)  #4维张量
print(a.view(4,28*28).shape)   #将（1,28，28）/后面3维合并为1维 输出结果为torch.Size([4,784])
#增加维度
print(a.unsqueeze(0).shape) #在最前面增加一个维度 输出 torch.Size([1,4,1,28,28])
#拷贝
print(a.repeat(4, 4, 1, 1).shape) #相当于第一个维度拷贝了4次，第二维度拷贝了4次，后面2个都是1次，结尾为（【16,4,28,28】）

#转置.t()
a = torch.randn(3,4)  #获取3行4列随机数
print(a.t().shape)    #输出4行3列数据,仅适用于2维
#交换维度
a = torch.rand(4,1,28,28)
print(a.transpose(0,3).shape) #将第1个和第4个维度进行交换，输出结果为torch.Size([28, 1, 28, 4])
print(a.permute(0,2,3,1).shape)  #将第1放在第1位置，第2放第4个位置，第3放第2，第4放第3位置，结果torch.Size([4, 28, 28, 1])
#合并
a = torch.rand(4, 32, 8)
b = torch.rand(5, 32, 8)
print(torch.cat([a, b], dim = 0).shape) #表示第一个维度进行合并，输出torch.Size([9, 32, 8])

a = torch.rand(32, 8)
b = torch.rand(32, 8)
c = torch.stack([a, b],dim=0)
print(c.shape)  #相当于2个平面叠加后高度为2的三维 结果为torch.Size([2, 32, 8])
#拆分
aa, bb = c.split([1, 1], dim=0)
print(aa.shape, bb.shape)  #在0维度拆分长度各位1，所以结果为torch.Size([1, 32, 8]) torch.Size([1, 32, 8])
a = torch.rand(4, 32, 8)
aa, bb= a.chunk(2, dim=0)
print(aa.shape, bb.shape) #拆分0维度，直接对半分,放回结果torch.Size([2, 32, 8]) torch.Size([2, 32, 8])

#数学运算
torch.add(a, b)  #加  或直接使用+
torch.sub(a, b)   #减  或直接使用-
torch.mul(a, b)   #乘   或直接使用*
torch.div(a, b)   #除   或直接使用/
a = a.pow(2)    #2次方   或直接使用**2
a = a.sqrt()    #开方

#矩阵相乘 用matmul
a = torch.tensor([
    [3, 3],
    [3, 3]
])
b = torch.tensor([
    [1, 1],
    [1, 1]
])
print(torch.matmul(a,b))   #结果为tensor([[6, 6], [6, 6]])
#4维矩阵相乘 只取后2维进行计算
a = torch.rand(4, 3, 28, 64)
b = torch.rand(4, 3, 64, 32)
print(torch.matmul(a, b).shape) #输出结果torch.Size([4, 3, 28, 32])

#范数 norm-p
a = torch.tensor([
    [3., 3.],
    [3., 3.]
])
b = torch.tensor([
    [1., 1.],
    [1., 1.]
])
print(a.norm(1), b.norm(1))  #求的ab的1范数，各元素绝对值之和，返回tensor(12.) tensor(4.)
print(a.norm(2), b.norm(2))  #求的ab的2范数，各元素平方和之后开根号，返回tensor(6.) tensor(2.)
print(a.norm(1,dim=0))    #在a的0维度上求1范数， a的0维度大小（【2,2】），0维度上即为【2】，[3,3]的1范数返回tensor([6., 6.])

a.min()  #求最小值
a.mean()   #求均值
a.sum()    #求累加
a.max()    #求最大值
a.prod()    #求累乘
a.argmax()   #返回回最大值的索引
a.argmin()    #返回最小值的索引
