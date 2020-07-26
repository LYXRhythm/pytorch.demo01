#梯度求导
import  torch
from    torch import autograd

x = torch.tensor(1.)                      #x值赋予1
a = torch.tensor(1., requires_grad=True)  #a的值赋予1，告知要对a求偏导
b = torch.tensor(2., requires_grad=True)  #b的值赋予2，告知要对b求偏导
c = torch.tensor(3., requires_grad=True)  #c的值赋予3，告知要对c求偏导

y = a**2 * x + b * x + c          #y的表达式

print('before:', a.grad, b.grad, c.grad)    #求导前abc的梯度信息
grads = autograd.grad(y, [a, b, c])
print('after :', grads[0], grads[1], grads[2])   #grad[0][1][2]分别表示对abc的偏导