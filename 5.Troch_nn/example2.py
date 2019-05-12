'''

'''
import numpy as np
import torch
from torch import nn
from torch import optim

N, D_in, H, D_out = 64, 1000, 100, 10
LEARNING_RATE=1e-4

# 随机创建一些训练数据
x=torch.randn(size=(N,D_in))
y=torch.randn(size=(N,D_out))

class MLP(nn.Module):
    def __init__(self,num_in,num_hidden,num_out):
        super(MLP,self).__init__()
        #sub modules
        self.linear1=nn.Linear(in_features=num_in,out_features=num_hidden,bias=False)
        self.linear2=nn.Linear(in_features=num_hidden,out_features=num_out,bias=False)
        self.relu=nn.ReLU()         #activation function


    def forward(self, x):
        h1=self.relu(self.linear1(x))
        h2=self.linear2(h1)
        return h2


#convient tools
mlp_model=MLP(num_in=D_in,num_hidden=H,num_out=D_out)
mse=nn.MSELoss(reduction="sum")     #loss
sgd=optim.SGD(params=mlp_model.parameters(),lr=LEARNING_RATE)

for it in range(500):
    if torch.cuda.is_available():
        mlp_model.cuda()
        x_cuda=x.cuda()
        y_cuda=y.cuda()
        # Forward pass
        y_pred = mlp_model.forward(x_cuda)  # mlp_model(x)
        # compute loss
        loss = mse(y_pred, y_cuda)
        print(it, loss.cpu().detach().numpy())

        # Backward pass
        # compute the gradient
        sgd.zero_grad()
        loss.backward()

        # update parameters of mlp_model
        sgd.step()
    else:
        # Forward pass
        y_pred = mlp_model.forward(x)  # mlp_model(x)
        # compute loss
        loss = mse(y_pred, y)
        print(it, loss.detach().numpy())

        # Backward pass
        # compute the gradient
        sgd.zero_grad()
        loss.backward()

        # update parameters of mlp_model
        sgd.step()





