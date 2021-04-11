import numpy as np
import torch
import torchvision as t


# 通过torch.tensor 创建张量
# data: 数据，可以使list，numpy
# dtype:数据类型，默认与data一致
# device:所在设备 cuda / cpu
# requires_grad : 是否需要梯度
# pin_memory : 是否存于锁页内存

# flag = True
flag = False

if flag:
    arr = np.ones((3,3))
    print("ndarray的数据类型:",arr.dtype)

    d = torch.tensor(arr,device='cuda')
    # t = torch.tensor(arr)
    print(d)


# flag = True
flag = False
# 通过numpy创建张量
if flag:
    arr = np.array([[1,2,3],[4,5,6]])
    ten = torch.from_numpy(arr)
    print("numpy array:",arr)
    print("tensor:",ten)

#依据数值创建
# flag = True
flag = False
if flag:
   out_t = torch.tensor([1])

   t = torch.zeros((3,3),out=out_t)
   print(t,'\n',out_t)
   print(id(t),id(out_t),id(t) == id(out_t))

#通过形状输出0张量
flag  = True
# flag = False
if flag:
    out_t = torch.tensor([1])
    in_put = np.array([[1,2,3],[4,5,6]])
    inputTen = torch.from_numpy(in_put)
    # t = torch.zeros_like(inputTen)
    # t = torch.ones((3,2))
    # t = torch.zeros_like((3.3),out = out_t)
    # t = torch.full((3,3),6)
    # t = torch.arange(2,20,5)
    # t = torch.linspace(2,60,5)
    t = torch.eye(6,5)
    print(t,'\n',out_t)
    print(id(t),id(out_t),id(t) == id(out_t))
