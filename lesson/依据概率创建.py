import torch


#均值：张量   标准差：张量
mean = torch.arange(1,5,dtype=torch.float)
std = torch.arange(1,5,dtype=torch.float)
t_normal = torch.normal(mean,std)
print("mean:{}\nstd:{}".format(mean,std))
print(t_normal)

#均值：标量   标准差：标量

t_normal2 = torch.normal(0.,1.,size=(4,))
print(t_normal2)

#均值：张量   标准差：标量
mean = torch.arange(1,5,dtype=torch.float)
std = 1
t_normal = torch.normal(mean,std)
print("mean:{}\nstd:{}".format(mean,std))
print(t_normal)

#标准正态分布
print("标准正态分布")
print(torch.randn((3,3)))
print("标准正态分布按形态生成")

t = torch.tensor([[1.,2.],[2,3]])
print("标准正态分布整数")
print(torch.randint(6,(3,3)))
print("标准正态分布整数按形态生成")
print(torch.randint_like(t,4,6))

print("生成0~n-1的随机排列")
print(torch.randperm(6))

print("以input为概率，生成伯努利分布（0,1）分布")
t = torch.normal(0.5,0.4,size=(4,))
print(t)
ma = t.gt(0)
print(ma)

ma = ma.lt(1)
print(ma)

print(torch.bernoulli(t))

