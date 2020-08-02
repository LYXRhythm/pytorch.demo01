#Softmax and crossentropy
import torch
import torch.nn as nn
import numpy as np

#定义softmax公式  axis=0表示从这一行取值
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])    #先输入一些数测试softmax的输出结果
outputs = softmax(x)
print('softmax numpy:', outputs)

x = torch.tensor([2.0, 1.0, 0.1])   #然后是测试了softmax类型
outputs = torch.softmax(x, dim=0)
print('softmax torch:', outputs)

def cross_entropy(actual, predicted):      #定义了交叉熵损失函数，传入的参数为实际值和与预测值
    EPS = 1e-15
    predicted = np.clip(predicted, EPS, 1 - EPS)   #clip将传入的预测值范围限定在1e-15到1 - 1e-15之间，1e-15小的归为1e-15
    loss = -np.sum(actual * np.log(predicted))
    return loss

Y = np.array([1, 0, 0])    #先定义了一个真实值和2两个预测值，验证一下刚刚定义好的交叉熵损失函数计算的效果
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')

loss = nn.CrossEntropyLoss()  #当然我们nn网络里面可以直接调用现成的交叉熵损失，

Y = torch.tensor([0])
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])  #传入交叉熵
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])
l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(f'PyTorch Loss1: {l1.item():.4f}')
print(f'PyTorch Loss2: {l2.item():.4f}')

# get predictions  troch.max()[1]只返回最大值的每个索引
_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(f'Actual class: {Y.item()}, Y_pred1: {predictions1.item()}, Y_pred2: {predictions2.item()}')

Y = torch.tensor([2, 0, 1])

Y_pred_good = torch.tensor(    #计算二维张量的损失值
    [[0.1, 0.2, 3.9],  # predict class 2
     [1.2, 0.1, 0.3],  # predict class 0
     [0.3, 2.2, 0.2]])  # predict class 1

Y_pred_bad = torch.tensor(
    [[0.9, 0.2, 0.1],
     [0.1, 0.3, 1.5],
     [1.2, 0.2, 0.5]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)
print(f'Batch Loss1:  {l1.item():.4f}')
print(f'Batch Loss2: {l2.item():.4f}')

# get predictions
_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(f'Actual class: {Y}, Y_pred1: {predictions1}, Y_pred2: {predictions2}')

# Binary classification
class NeuralNet1(nn.Module):    #定义网络结构
    def __init__(self, input_size, hidden_size):  #传入如输入参数和隐藏层的大小
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)   #用于设置网络全连接层
        self.relu = nn.ReLU()  #进行relu变换
        self.linear2 = nn.Linear(hidden_size, 1)  #再经过一次全连接层输出结果为1层

    def forward(self, x):   #定义了向前传播的过程
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # sigmoid at the end
        y_pred = torch.sigmoid(out)
        return y_pred

model = NeuralNet1(input_size=28 * 28, hidden_size=5)   #完成网络模型的建立
criterion = nn.BCELoss()   #调用交叉熵损失

# Multiclass problem   与上面一样 就多了个参数而已，作为最后后的输出维度
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # no softmax at the end
        return out


model = NeuralNet2(input_size=28 * 28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss()  # (applies Softmax)
