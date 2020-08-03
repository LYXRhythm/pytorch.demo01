#向前神经网络
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Device configuration  启动gpu加速
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters   设置输入参数
input_size = 784  # 28x28 输入784
hidden_size = 500  #隐藏层大小为500
num_classes = 10   #最后输出结果10分类
num_epochs = 2    #设置常数
batch_size = 100     #每次投喂的数据量
learning_rate = 0.001  #学习速率

# MNIST dataset   加载手写数据集 将数据放入数据库中
train_dataset = torchvision.datasets.MNIST(root='./mnist_data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./mnist_data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader 从数据库中获取手写数据集 每次获得batch_size个样本
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

examples = iter(test_loader) #这个iter能把一个batch_size提取出来
example_data, example_targets = examples.next()   #获得特征值和标签值

for i in range(6):  #展示出前6个的手写数字图
    plt.subplot(2, 3, i + 1)
    plt.imshow(example_data[i][0], cmap='gray')
plt.show()


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):   #定义神经网络的类
    def __init__(self, input_size, hidden_size, num_classes):  #输入传出传入层参数大小
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)  #全连接层
        self.relu = nn.ReLU()  #relu
        self.l2 = nn.Linear(hidden_size, num_classes)  #全连接层

    def forward(self, x):    #向前传播过程
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out


model = NeuralNet(input_size, hidden_size, num_classes).to(device)   #完成向前传播模型建立并使用cuda加速

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  #定义损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  #定义学习率和损失函数

# Train the model
n_total_steps = len(train_loader)    #获得训练集数据长度
for epoch in range(num_epochs):     #建立循环
    for i, (images, labels) in enumerate(train_loader): #枚举数据
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        images = images.reshape(-1, 28 * 28).to(device)   #获得图片尺寸cuda加速
        labels = labels.to(device)   #获得标签

        # Forward pass
        outputs = model(images)    #传入向前传播模型
        loss = criterion(outputs, labels)   #调用损失函数计算损失值

        # Backward and optimize
        optimizer.zero_grad()  #关闭梯度
        loss.backward()   #向后传播
        optimizer.step()  #调用优化器

        if (i + 1) % 100 == 0:   #每100个数据进行一次输出
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():  #关闭梯度
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:   #获得测试集数据
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)   #调用模型
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)   #索引最大值
        n_samples += labels.size(0)   #放回样本数
        n_correct += (predicted == labels).sum().item()   #计算预测准确个数

    acc = 100.0 * n_correct / n_samples    #获得准确率 并输出
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
