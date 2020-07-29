import  torch
from    torch import nn
from    torch.nn import functional as F

class Lenet5(nn.Module):  #定义Lenet网络结构类
    """
    for cifar10 dataset.
    """
    def __init__(self):
        super(Lenet5, self).__init__()  #调用类的方法初始化父类
        self.conv_unit = nn.Sequential(   #nn.Sequential包含很多子类，快速上手卷积网络
            # x: [b,3,32,32] => [b,16,28,28 ]，先卷积 输入3层，输出16层，卷积核大小为5*5，步长为1
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=0),
            #[b,16,14,14]第二层经过池化，卷积核大小为2*2步长为2，层数还是16，输出形状变为1/4,14*14
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            #[b,32,10,10]第三层又是经过卷积，16层输入，32层输出
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),
            #[b,32,5,5]第四层经过池化，32层输出，输出形状变为1/4,8*8
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        # # [b, 3, 32, 32] 以下代码可以用于计算结果卷积层输入与输出的维度
        # tmp = torch.randn(2, 3, 32, 32)
        # out = self.conv_unit(tmp)
        # # [b, 32, 5, 5]
        # print('conv out:', out.shape)

        # flatten
        # fc unit 定义全连接层
        self.fc_unit = nn.Sequential(
            nn.Linear(32*5*5, 32),    #从卷积网络输出到全连接层的数据是已经平铺为1维了
            nn.ReLU(), #使用激活函数ReLU
            nn.Linear(32, 10) #输入32层，输出10层
        )

        # # use Cross Entropy Loss
        # self.criteon = nn.CrossEntropyLoss()
    def forward(self, x): #定义前向传播，后向传播不需要定义，自动生成

        batchsz = x.size(0)   #相当于获取图片张数
        # [b, 3, 32, 32] => [b, 16, 5, 5]
        x = self.conv_unit(x)   #将输入图片传入卷积网络
        # [b, 16, 5, 5] => [b, 16*5*5]
        x = x.view(batchsz, 32*5*5)  #平铺为一维
        # [b, 16*5*5] => [b, 10]
        logits = self.fc_unit(x)   #经过全连接层
        # # [b, 10]
        # pred = F.softmax(logits, dim=1) 分类函数
        # loss = self.criteon(logits, y) 计算损失
        return logits
def main():
    net = Lenet5()
    tmp = torch.randn(2, 3, 32, 32)  #输入数据
    out = net(tmp)  #传进lenet网络
    print('lenet out:', out.shape)  #打印输出结果
if __name__ == '__main__':
    main()