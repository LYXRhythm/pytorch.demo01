#迁移学习
import torch
import torch.nn as nn
import torch.optim as optim  #导入上面的torch.nn包之后还需导入torch.optim包，定义损失函数和优化方法
from torch.optim import lr_scheduler   #提供了基于多种epoch数目调整学习率的方法.
import numpy as np
import torchvision
from torchvision import datasets, models, transforms   #调用导入数据包、模型下载、转换
import matplotlib.pyplot as plt
import time
import os   #导入标准库os利用其中的API
import copy   #浅复制

mean = np.array([0.5, 0.5, 0.5])   #均值和方差 后面直接公式计算
std = np.array([0.25, 0.25, 0.25])

# 训练数据集需要扩充和归一化
# 验证数据集仅需要归一化
data_transforms = {            #对数据做变换
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),   #输入的图片需要分辨率为224*224，所以需要裁剪。
        transforms.RandomHorizontalFlip(),   #图片做垂直翻转
        transforms.ToTensor(),
        transforms.Normalize(mean, std)    #正则化处理
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

#图像保存路径
data_dir = 'hymenoptera_data'
#查看对应文件夹的labels  存在数据库 输出结果是：{'bees': 1, 'ants': 0}
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x])for x in ['train', 'val']}
#对数据库中的存入内容进行输出，每次输出4个，顺序打乱，不加进程
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,shuffle=True, num_workers=0)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}  #输出训练集和测试集的图片数
class_names = image_datasets['train'].classes   #训练集的名字蜜蜂或蚂蚁
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  #判断能否使用gpu加速
print(class_names) #打印输出名字

def imshow(inp, title):   #定义图片输出函数
    inp = inp.numpy().transpose((1, 2, 0))   #调换数据通道位置
    inp = std * inp + mean     #反归一化
    inp = np.clip(inp, 0, 1)    #设置范围
    plt.imshow(inp)      #打印输出图片
    plt.title(title)     #输出标题
    plt.show()

# Get a batch of training data 获得训练集的数据
inputs, classes = next(iter(dataloaders['train']))
# Make a grid from batch 将多个图片拼成一张图片
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes]) #显示图片和各图片的标题

# 训练模型函数，参数scheduler是一个 torch.optim.lr_scheduler 学习速率调整类对象  这里定义模型 共输入了5个参数
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()   #开始计时

    best_model_wts = copy.deepcopy(model.state_dict())  #调用深拷贝，model.state_dict()保存了每一组交叉验证模型的参数
    best_acc = 0.0

    for epoch in range(num_epochs):   #调用循环输出num_epochs次
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 每个epoch都有一个训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 训练模式
            else:
                model.eval()   # 验证模式

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:  #获得训练集或测试集的数据
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 训练阶段开启梯度跟踪
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)       #调用模型 这个model后面定义为resnet18
                    _, preds = torch.max(outputs, 1)   #获得输出值最大索引下标
                    loss = criterion(outputs, labels)   #调用优化器

                    # 仅在训练阶段进行后向+优化
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # 统计    计算总的损失值和准确数
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]   #平均损失值和准确率
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 记录最好的状态 使测试集直接调用
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
#记录时间
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 返回最佳参数的模型
    model.load_state_dict(best_model_wts)
    return model

model = models.resnet18(pretrained=True)  #models导入resnet18网络
num_ftrs = model.fc.in_features   #获得全连接层的输入channel个数
model.fc = nn.Linear(num_ftrs, 2)   #用这个channel个数和你要做的分类类别数（这里是2）替换原模型的全连接层
model = model.to(device)

# 使用分类交叉熵 Cross-Entropy 作损失函数，动量SGD做优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
#定义学习率的变化策略，表示每隔step_size个epoch就将学习率降为原来的gamma倍
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=25) #调用训练模型

model_conv = torchvision.models.resnet18(pretrained=True) #程序自动下载已经训练好的参数
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()


optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)
