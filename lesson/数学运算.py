import torch

#加减乘除
flag = True
#flag = False
if flag:
    t_0 = torch.randn((3,3))
    t_1 = torch.ones_like(t_0)
    t_add = torch.add(t_0,10,t_1)
    print("t_0:\n{}\nt_1:\n{}\nt_add:\n{}".format(t_0,t_1,t_add))

    t_div = torch.div(t_0,10)
    print("t_0:\n{}\nt_div:\n{}".format(t_0,t_div))

    t_mul = torch.mul(t_0,10)
    print("t_0:\n{}\nt_mul:\n{}".format(t_0,t_mul))
