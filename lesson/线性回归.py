# -*- coding:utf-8 -*-
"""
@file name : Linear-Regression
@author    : chankequan
@date      : 2021-04-11
@brief     : 一元线性回归模型
"""
import torch
import matplotlib.pyplot as plt

torch.manual_seed(10)
#学习率
lr = 0.0001

#创建训练数据

x = torch.rand(20, 1)*10  #data(tensor),shape = (20,1)
# 偏执为5 再加些噪声
y = 2 * x + (5 + torch.randn(20, 1))  # y data (tensor) shape = (20,1)

#构建线性回归参数   张量 梯度求导
w = torch.randn((1),requires_grad=True)    #w正态分布初始化
b = torch.zeros((1),requires_grad=True)    # b 初始化为0

for iteration in range(1000):
    #向前传播
    wx = torch.mul(w,x)
    y_pred = torch.add(wx,b)

    #计算 MES（均方差） loss  mean求均值， 因y 为tensor,故可使用求均值方法
    loss = (0.5 *(y - y_pred) ** 2).mean()

    #反向传播
    loss.backward()

    #更新参数
    #PyTorch网站上不要使用就地操作.除非在沉重的内存压力下工作,否则在大多数情况下,不使用就地操作会更有效率.
    # 其次, 在使用就地操作时可能会出现计算梯度的问题
    #就地操作
    # b.data.sub_(lr * b.grad)
    # w.data.sub_(lr * w.grad)
    # 非就地操作
    b.data = b.data.sub_(lr * b.grad)
    w.data = w.data.sub_(lr * w.grad)

    # 绘图
    if iteration % 20 == 0:
        #scatter 绘制散点图
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)
        plt.text(2,20,'Loss=%.4f' % loss.data.numpy(),fontdict={'size': 20, 'color': 'red'})
        plt.xlim(1.5, 10)
        plt.ylim(8, 28)
        plt.title("Iteration:{}\n w:{}\nb:{}".format(iteration,w.data.numpy(),b.data.numpy()))
        plt.pause(0.5)

        if loss.data.numpy() < 0.6:
            break