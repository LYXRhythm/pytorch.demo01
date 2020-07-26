#对比cpu与gpu加速
import torch
import time

print(torch.__version__)           #查看版本
print(torch.cuda.is_available())       #能否运行gpu版本cuda
# print('hello, world.')
a = torch.randn(10000, 1000)
b = torch.randn(1000, 2000)
t0 = time.time()
c = torch.matmul(a, b)   #计算两个张量相乘所需时间
t1 = time.time()
print(a.device, t1 - t0, c.norm(2))       #计算cpu运行时间

device = torch.device('cuda')
a = a.to(device)
b = b.to(device)
t0 = time.time()
c = torch.matmul(a, b)
t2 = time.time()
print(a.device, t2 - t0, c.norm(2))      #计算初始化gpu运行的时间

t0 = time.time()
c = torch.matmul(a, b)
t2 = time.time()
print(a.device, t2 - t0, c.norm(2))       #计算gpu运行时间

