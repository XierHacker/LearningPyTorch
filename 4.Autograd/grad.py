import numpy
import torch
#import torch

#like Variable
a=torch.tensor([[1,2,3,4],[5,6,7,8]],dtype=torch.float32,requires_grad=True)
print(a)

#结果v_out要是标量才能够反向传播,v_out=1/8*sum(a*a)
#
v_out=torch.mean(a*a)

v_out.backward()
print(a.grad)
