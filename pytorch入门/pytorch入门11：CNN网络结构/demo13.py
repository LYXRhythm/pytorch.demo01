#CNN网络
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device configuration  cuda加速
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 5
batch_size = 4
learning_rate = 0.001

# 第一步：准备数据
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
train_dataset = torchvision.datasets.CIFAR10(root='./cifar', train=True,
                                             download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./cifar', train=False,
                                            download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False)
#10分类
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):   #打印出图片
    img = img / 2 + 0.5  # 反归一化
    npimg = img.numpy()   #将tensor转化成numpy类型
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  #在pytorch中图片的格式是（通道数，行像素、列像素），而numpy通道数在后面
    plt.show()


# get some random training images
dataiter = iter(train_loader)  #然后加载了训练集的数据，将训练集中的特征值和标签分别赋予images和labels.
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))

#第二步cnn网络模型建立
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)    #第一层是卷积传入3表示图片RGB三层通道。然后输出为6层。卷积核大小为5*5大小
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) #将值平铺成一维，输入全连接层，输出数据变成120
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):  #前向传播
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = x.view(-1, 16 * 5 * 5)  # -> n, 400
        x = F.relu(self.fc1(x))  # -> n, 120
        x = F.relu(self.fc2(x))  # -> n, 84
        x = self.fc3(x)  # -> n, 10
        return x

model = ConvNet().to(device)  #完成CNN网络模型建立，调用cuda加速

#第三步选择损失函数和优化器
criterion = nn.CrossEntropyLoss() #交叉熵损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  #随机梯度下降

#第四步 训练循环网络
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 2000 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

# print('Finished Training')
# PATH = './cnn.pth'
# torch.save(model.state_dict(), PATH)

with torch.no_grad():          #打印准确率
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)   #返回最大索引
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()  #计算准确数

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1  #计算准确数
            n_class_samples[label] += 1   #计算总样本数

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')

