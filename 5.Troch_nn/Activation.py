import numpy as np
import torch
import matplotlib.pyplot as plt

x=np.linspace(start=-50,stop=50,num=1000)

input=torch.tensor(x,dtype=torch.float32,requires_grad=True)

#print(input)
relu=torch.nn.ReLU()
output=relu(input)
print(output)

plt.plot(x,output.detach().numpy())
plt.show()