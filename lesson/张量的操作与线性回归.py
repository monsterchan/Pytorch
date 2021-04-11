import torch

# flag = True
flag = False
if flag:
    t = torch.ones((2,3))

    t_0 = torch.cat([t,t],dim=0)
    t_1 = torch.cat([t,t],dim=1)
    t_2 = torch.stack([t,t],dim=0)
    t_4 = torch.stack([t,t],dim=1)
    t_3 = torch.stack([t,t],dim=2)
    print("t_0:{} shape:{}\n t_1:{} shape:{}".format(t_0,t_0.shape,t_1,t_1.shape))
    print("t_2:{} shape:{}\n t_3:{} shape:{}".format(t_2,t_2.shape,t_3,t_3.shape))
    print("t_4:{} shape:{}\n t_3:{} shape:{}".format(t_4,t_4.shape,t_3,t_3.shape))

# flag = True
flag =False
# 张量的切分
if flag:
    a = torch.ones(2,5)
    list_of_tensor = torch.chunk(a,dim=1,chunks=2)
    # 先整除，向上取整，最后一个剩多少取多少
    for idx, t in enumerate(list_of_tensor):
        print("第{}个张量:{},shape is {}".format(idx+1, t, t.shape))

# flag = True
flag =False
# 张量的切分 split
if flag:
    # t = torch.ones((2, 5))
    t = torch.ones((2, 7))
    list_of_tensor = torch.split(t,3,dim=1)
    for idx, t in enumerate(list_of_tensor):
        print("第{}个张量:{},shape is {}".format(idx + 1, t, t.shape))

# flag = True
flag =False
# 张量的索引 index_select
# 索引的index都是long型，如果是float会报错
if flag:
    t = torch.randint(0, 9, size=(3,3))
    idx = torch.tensor([0,2],dtype=torch.long)
    t_select = torch.index_select(t,dim=0,index=idx)
    print("t:\n{}\nt_select:\n{}".format(t,t_select))

# flag = True
flag =False
# 张量的索引 masked_select
if flag:
    t = torch.randint(0,9,size=(3,3))
    mask = t.gt(5)
    t_select = torch.masked_select(t,mask)
    print("t:\n{}\nmask:\n{}\nt_select:\n{}".format(t,mask,t_select))

# flag = True
flag =False
#张量变换 reshape
if flag:
    t = torch.randint(0,9,size=(3,4))
    t_reshape =torch.reshape(t,(2,-1))
    print("t:{}\nt_reshape:\n{}".format(t,t_reshape))

# flag = True
flag =False
#张量变换 transpose 交换张量的两个维度
if flag:
    t = torch.rand((2,3, 4))
    t_transpose = torch.transpose(t,dim0=1,dim1=2)
    print("t:{} shape:{}\nt_transpose:\n{} shape:{}".format(t,t.shape, t_transpose,t_transpose.shape))
    d_t = torch.rand(2,6)
    t_t = torch.t(d_t)
    print("d_t:{} shape:{}\nt_t:\n{} shape:{}".format(d_t,d_t.shape, t_t,t_t.shape))

flag = True
# flag =False
#张量变换 squeeze
if flag:
    t = torch.rand((1,2,3,1))
    t_sq = torch.squeeze(t)
    # 未指定维度，压缩最外层两边为维度为1的轴
    t_0 = torch.squeeze(t,dim=0)
    t_1 = torch.squeeze(t,dim=1)
    t_2 = torch.squeeze(t,dim=3)
    print(t.shape)
    print(t_sq.shape)
    print(t_0.shape)
    print(t_1.shape)
    print(t_2.shape)