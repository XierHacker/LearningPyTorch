import numpy as np
import torch
import matplotlib.pyplot as plt

#data generate
X=torch.unsqueeze(input=torch.linspace(start=-20,end=20,steps=80),dim=1).cuda()
y=2*X+4*torch.randn(size=X.size()).cuda()          #add noise

#plt.plot(X.numpy(),y.numpy(),"r+")
#plt.show()

#define model
class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression,self).__init__()
        self.fc=torch.nn.Linear(in_features=1,out_features=1)

    def forward(self, input):
        logits=self.fc(input)
        return logits


#training process
def train():
    # use model
    if torch.cuda.is_available():
        model = LinearRegression().cuda()
    else:
        model = LinearRegression()

    #loss function and optimizer
    MSE_loss=torch.nn.MSELoss()
    adam=torch.optim.Adam(params=model.parameters(),lr=0.1)

    max_epochs=100
    for i in range(max_epochs):
        # forward
        logits = model.forward(input=X)
        loss = MSE_loss(logits, y)

        # backward
        adam.zero_grad()
        loss.backward()
        adam.step()

        #log
        if (i+1)%10==0:
            print(i+1,":",loss.item())
            plt.plot(X.cpu().numpy(),y.cpu().numpy(),"r+")
            plt.plot(X.cpu().numpy(),logits.cpu().detach().numpy())
            plt.show()


if __name__=="__main__":
    train()