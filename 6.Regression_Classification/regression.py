import numpy as np
import torch
import matplotlib.pyplot as plt

#data generate
X=torch.unsqueeze(input=torch.linspace(start=-20,end=20,steps=80),dim=1)
y=2*X+4*torch.randn(size=X.size())          #add noise

#plt.plot(X.numpy(),y.numpy(),"r+")
#plt.show()


class LinearRegression(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self, *input):
        pass
    