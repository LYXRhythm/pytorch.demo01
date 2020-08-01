#Dataset and DataLoader
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset): #定义加载酒数据集类

    def __init__(self):    #加载数据集赋予x和y
        # read with numpy or pandas
        xy = np.loadtxt('./wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, 1:])  # size [n_samples, n_features]
        self.y_data = torch.from_numpy(xy[:, [0]])  # size [n_samples, 1]

    # support indexing such that dataset[i] can be used to get i-th sample  检索x、y中元素
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size 获取样本数
    def __len__(self):
        return self.n_samples

# create dataset  完成酒类的定义赋予dataset
dataset = WineDataset()
#调用DataLoader函数，第一个参数把dataset获得数据赋给dataloader，第二个参数为每次投喂的是数据大小。第三参数要打乱数据，第四个参数为多进程个数。
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)

# Dummy Training loop
total_samples = len(dataset)   #获得样本数
n_iterations = math.ceil(total_samples / 4)   #向上取整
print(total_samples, n_iterations)

for i, (inputs, labels) in enumerate(dataloader):
    #forward backward, update  每隔5次看一下输出的数据
    if (i+1)% 5 == 0:
        print(f'Step {i+1}/{n_iterations}| Inputs {inputs.shape} | Labels {labels.shape}')

# some famous datasets are available in torchvision.datasets
# e.g. MNIST, Fashion-MNIST, CIFAR10, COCO

# train_dataset = torchvision.datasets.MNIST(root='./mnist_data',
#                                            train=True,
#                                            transform=torchvision.transforms.ToTensor(),
#                                            download=True)
#
# train_loader = DataLoader(dataset=train_dataset,
#                           batch_size=3,
#                           shuffle=True)
# # look at one random sample
# dataiter = iter(train_loader)
# data = dataiter.next()
# inputs, targets = data
# print(inputs.shape, targets.shape)
