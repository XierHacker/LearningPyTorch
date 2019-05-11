'''
进行一个简单线性拟合的例子,使用pytorch的tensor，自动求导，手动更新
本质就是将numpy ndarray换成了tensor来用
'''

import numpy as np
import torch

N, D_in, H, D_out = 64, 1000, 100, 10
LEARNING_RATE=1e-6


# 随机创建一些训练数据
x=torch.randn(size=(N,D_in))
#print("x:\n",x.shape)
y=torch.randn(size=(N,D_out))
#print("y:\n",y.shape)

#weights
w1=torch.randn(size=(D_in,H),requires_grad=True)
w2=torch.randn(size=(H,D_out),requires_grad=True)

for it in range(500):
    # Forward pass
    h=torch.mm(x,w1)
    h_relu=torch.clamp(h,min=0.0)
    y_pred=torch.mm(h_relu,w2)

    # compute loss
    loss = (y_pred - y).pow(2).sum()
    print(it, loss.detach().numpy())

    # Backward pass
    # compute the gradient
    loss.backward()

    # update weights of w1 and w2
    with torch.autograd.no_grad():
        w1 -= LEARNING_RATE * w1.grad
        w2 -= LEARNING_RATE * w2.grad
        w1.grad.zero_()
        w2.grad.zero_()