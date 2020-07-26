#简单回归案例
import numpy as np

# y = wx + b
def compute_error_for_line_given_points(b, w, points): #定义求误差平方和函数
    totalError = 0         #赋予初始值为0 用于计算所有的误差平方和
    for i in range(0, len(points)):  #获得数据集中的第一列为x，第二列为y
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2       #求得所有误差平方和之和 公式loss = (y-(w*x+b))^2
    return totalError / float(len(points))       #返回平均后的误差平方和

def step_gradient(b_current, w_current, points, learningRate): #定义计算b与w梯度值
    b_gradient = 0         #初始值0
    w_gradient = 0
    N = float(len(points))     #获取数据总长度
    for i in range(0, len(points)):   #获得第一列给x，第二列给y
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((w_current * x) + b_current))  #计算b对loss偏导后的值，2(((w*x)+b)-y)
        w_gradient += -(2/N) * x * (y - ((w_current * x) + b_current))  #计算w对loss偏导后的值，2x(((w*x)+b)-y)
    new_b = b_current - (learningRate * b_gradient)     #更新b的值
    new_w = w_current - (learningRate * w_gradient)    #更新w的值
    return [new_b, new_w]   #返回更新后的b与w

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations): #定义循环迭代函数
    b = starting_b   #赋予b初始值
    m = starting_m   #赋予m初始值
    for i in range(num_iterations):    #循环迭代num次
        b, m = step_gradient(b, m, np.array(points), learning_rate)   #开始梯度下降
    return [b, m] #返回迭代完后的b与m

def run():  #定义主代码
    points = np.genfromtxt("data.csv", delimiter=",")    #读取数据，逗号为分隔符
    learning_rate = 0.0001      #学习率为0.0001
    initial_b = 0 # 赋予初始b值
    initial_w = 0 # 赋予初始w值
    num_iterations = 1000     #迭代1000次
    print("Starting gradient descent at b = {0}, w = {1}, error = {2}" #输出起始b,w和误差平方和
          .format(initial_b, initial_w,
                  compute_error_for_line_given_points(initial_b, initial_w, points))
          )
    print("Running...")
    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, w = {2}, error = {3}".          #输出迭代完后的b,w和误差平方和
          format(num_iterations, b, w,
                 compute_error_for_line_given_points(b, w, points))
          )

if __name__ == '__main__':      #运行代码
    run()