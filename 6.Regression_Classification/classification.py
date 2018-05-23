import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

'''
load data
'''

def load_mnist(path):
    train_frame = pd.read_csv(path+"train.csv")[:40000]
    valid_frame = pd.read_csv(path + "train.csv")[40000:]
    test_frame = pd.read_csv(path+"test.csv")

    y_train = train_frame.pop(item="label").values
    #print(y_train.shape)
    y_valid = valid_frame.pop(item="label").values
    #print(y_valid.shape)

    # trans format
    X_train = train_frame.astype(np.float32).values
    X_valid = valid_frame.astype(np.float32).values
    X_test = test_frame.astype(np.float32).values

    #to tensor
    return X_train,y_train,X_valid,y_valid,X_test


'''
model
'''

#parameters
WIDTH=28
HIGHT=28
NUM_1=300
NUM_2=100
NUM_OUT=10

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1=torch.nn.Sequential(
            torch.nn.Linear(in_features=WIDTH*HIGHT,out_features=NUM_1),
            #torch.nn.BatchNorm1d(num_features=NUM_1),
            torch.nn.ReLU(inplace=True)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=NUM_1, out_features=NUM_2),
            #torch.nn.BatchNorm1d(num_features=NUM_2),
            torch.nn.ReLU(inplace=True)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(in_features=NUM_2, out_features=NUM_OUT),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, input):
        logits_1=self.layer1(input)
        logits_2=self.layer2(logits_1)
        logits=self.layer3(logits_2)
        return logits


'''
train
'''
#parameters
MAX_EPOCHS=20
BATCH_SIZE=128
LEARNING_RATE=0.0002

def train(X,y):
    #model=MLP()
    #use model
    if torch.cuda.is_available():
        model = MLP().cuda()
    else:
        model = MLP()

    # loss function and optimizer
    CE_loss = torch.nn.CrossEntropyLoss()
    adam = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    for epoch in range(MAX_EPOCHS):
        print("epoch:",epoch+1)
        epoch_loss=[]
        epoch_accuracy=[]
        for j in range(40000//BATCH_SIZE):
            #forward
            logits=model.forward(input=X[j*BATCH_SIZE:(j+1)*BATCH_SIZE])
            prediction=torch.argmax(input=logits,dim=1)

            loss=CE_loss(logits,y[j*BATCH_SIZE:(j+1)*BATCH_SIZE])
            accuracy=accuracy_score(y_true=y.cpu()[j*BATCH_SIZE:(j+1)*BATCH_SIZE],y_pred=prediction.cpu().numpy())

            #backward
            adam.zero_grad()
            loss.backward()
            adam.step()

            epoch_loss.append(loss.item())
            epoch_accuracy.append(accuracy)

        print("----average loss:",sum(epoch_loss)/len(epoch_loss))
        print("----average accuracy:",sum(epoch_accuracy)/len(epoch_accuracy))


#test


if __name__=="__main__":
    print("Loading Data")
    X_train, y_train, X_valid, y_valid, X_test=load_mnist(path="../data/MNIST/")
    #trans to tensors
    X_train=torch.from_numpy(X_train).cuda()
    y_train = torch.from_numpy(y_train).cuda()

    #training
    print("Training Start!")
    train(X=X_train,y=y_train)
