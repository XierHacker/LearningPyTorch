'''
进行一个简单线性拟合的例子,使用pytorch的tensor，自动求导，自动更新
'''

import numpy as np
import torch
from torch import nn
from torch import optim

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

#convient tools
relu=nn.ReLU()                      #activation
mse=nn.MSELoss(reduction="sum")     #loss
sgd=optim.SGD(params=(w1,w2),lr=LEARNING_RATE)

for it in range(500):
    # Forward pass
    h=torch.mm(x,w1)
    h_relu=relu(h)
    y_pred=torch.mm(h_relu,w2)

    # compute loss
    loss = mse(y_pred,y)
    print(it, loss.detach().numpy())

    # Backward pass
    # compute the gradient
    sgd.zero_grad()
    loss.backward()

    # update weights of w1 and
    sgd.step()
