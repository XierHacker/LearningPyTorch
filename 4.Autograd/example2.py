'''
进行一个简单线性拟合的例子,使用pytorch的tensor，手动求导
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
w1=torch.randn(size=(D_in,H))
w2=torch.randn(size=(H,D_out))

for it in range(500):
    # Forward pass
    h=torch.mm(x,w1)
    h_relu=torch.clamp(h,min=0.0)
    y_pred=torch.mm(h_relu,w2)

    # compute loss
    loss = (y_pred - y).pow(2).sum()
    print(it, loss.numpy())

    # Backward pass
    # compute the gradient
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # update weights of w1 and w2
    w1 -= LEARNING_RATE * grad_w1
    w2 -= LEARNING_RATE * grad_w2
