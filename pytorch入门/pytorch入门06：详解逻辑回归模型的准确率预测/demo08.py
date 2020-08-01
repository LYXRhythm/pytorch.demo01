#简单的逻辑回归 预测癌细胞特征数据
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#0)prepare data
bc = datasets.load_breast_cancer()  #加载数据集  根据癌细胞的特征来判断良性/恶性
X,y = bc.data, bc.target  #然后是将癌细胞的特征数据赋给了X,标签数据赋予了y

n_sample, n_features = X.shape  #获得特征数据的行列数
print(f"特征数据的行有：{n_sample} 特征数据的列有：{n_features}")  #569行30列


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=1234)#划分数据集

#scale
sc = StandardScaler()  #x =(x - 𝜇)/𝜎 （样本-均值）/方差
X_train = sc.fit_transform(X_train)  #先拟合数据 在标准化
X_test = sc.transform(X_test)      #标准化数据

X_train = torch.from_numpy(X_train.astype(np.float32))  #将numpy数据转化为tensor数据
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)   #竖向排列
y_test = y_test.view(y_test.shape[0], 1)

#1) model
#f = wx+b, sigmoid at end
class LogisticRegression(nn.Module):    #定义逻辑回归
    def __init__(self, n_input_features):    #输入数据参数
        super(LogisticRegression, self).__init__()  #初始化
        self.linear = nn.Linear(n_input_features, 1)   #调用线性拟合

    def forward(self, x):   #向前传播过程
        y_predicted = torch.sigmoid(self.linear(x))   #sigmoid激活函数
        return y_predicted

model = LogisticRegression(n_features)  #完成模型建立

#2 loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss()  #交叉熵损失函数
optimizer = torch.optim.SGD(model.parameters(),lr =learning_rate)  #优化器选择随机梯度下降

#3 training loop
num_epochs = 10000
for epoch in range(num_epochs):
    #forword pass and loss
    y_predicted = model(X_train)      #预测y_
    loss = criterion(y_predicted, y_train)  #调用交叉熵损失函数

    #backward pass
    loss.backward()

    #updates
    optimizer.step()
    #zero gradients
    optimizer.zero_grad()   #向后传播不需要梯度，但向前传播需要梯度，不归零，下次向前传播累加梯度

    if(epoch+1)%10 == 0:
        print(f'epoch:{epoch+1},loss={loss.item():.4f}')  #打印数据

with torch.no_grad():   #关闭梯度，节约内存
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()  #返回四舍五入后的值
    acc = y_predicted_cls.eq(y_test).sum()/float(y_test.shape[0])  #计算准确率
    print(f'accuracy = {acc:4f}')  #打印最后的预测值准确率

