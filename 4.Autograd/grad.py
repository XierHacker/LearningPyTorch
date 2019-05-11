import numpy
import torch

x=torch.tensor(data=3.0,dtype=torch.float32,requires_grad=True)
y1=2*x*x


y1.backward()


print("x.grad:",x.grad)