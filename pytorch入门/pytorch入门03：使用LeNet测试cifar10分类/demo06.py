import  torch
from    torch.utils.data import DataLoader    #导入下载功能通道
from    torchvision import datasets      #加载数据使用
from    torchvision import transforms    #对数据做变换使用
from    torch import nn, optim   #导入nn网络和optim优化器

from    lenet5 import Lenet5     #引进类
# from    resnet import ResNet18

def main():
    batchsz = 128   #每次投喂的数据量
       #datasets加载CIFAR10数据集到本地，命名为cifar，transform对数据做变换，32*32的大小，自动下载数据集
    cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)  #每次导入batchsz那么多的数据
#定义测试集与训练集一样
    cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)


    x, label = iter(cifar_train).next()   #打印训练集数据和标签形状
    print('x:', x.shape, 'label:', label.shape)

    device = torch.device('cuda')   #调用cuda加速
    model = Lenet5().to(device)    #将进入的Lenet5也使用cuda加速

    criteon = nn.CrossEntropyLoss().to(device)     #调用损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-3)   #调用Adam优化器，
    print(model)   #打印类的实例

    for epoch in range(1000):
        model.train()  #变成训练模式
        for batchidx, (x, label) in enumerate(cifar_train):  #获取数据
            # [b, 3, 32, 32]
            x, label = x.to(device), label.to(device)  #cuda加速
            logits = model(x)  #通过lenet5训练
            # logits: [b, 10]   # label:  [b]
            # loss: tensor scalar
            loss = criteon(logits, label) #计算损失
            # backprop
            optimizer.zero_grad()  #优化器把梯度清零 防梯度累加
            loss.backward()
            optimizer.step()  #运行优化器走流程
        print(epoch, 'loss:', loss.item())  #打印每次损失，item表示转化成numpy类型


        model.eval()  #变成测试模式
        with torch.no_grad():   #这里告诉pytorch运算时不需计算图的
            # test
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:   #获取测试集数据
                # [b, 3, 32, 32]
                # [b]
                x, label = x.to(device), label.to(device)   #调用cuda

                # [b, 10]
                logits = model(x)
                # [b]
                pred = logits.argmax(dim=1)  #在第2个维度上索引最大的值的下标
                # [b] vs [b] => scalar tensor  比较预测值与真实值预测对的数量 eq是否相等
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)  #统计输入总数
                # print(correct)

            acc = total_correct / total_num  #计算平均准确率
            print(epoch, 'test acc:', acc)



if __name__ == '__main__':
    main()
