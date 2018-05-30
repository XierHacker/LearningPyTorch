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

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,out_channels=6,kernel_size=3,stride=1,padding=1),
            torch.nn.MaxPool2d(kernel_size=2,stride=2)
            #torch.nn.BatchNorm1d(num_features=NUM_1),
            #torch.nn.ReLU(inplace=True)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
            #torch.nn.ReLU(inplace=True)
        )
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=400, out_features=120),
            torch.nn.Linear(in_features=120,out_features=84),
            torch.nn.Linear(in_features=84,out_features=10)
            #torch.nn.Softmax(dim=1)
        )

    def forward(self, input):
        logits_1=self.conv1(input)
        logits_2=self.conv2(logits_1)
        #print(logits_2.size())
        logits_2=logits_2.view(logits_2.size(0),-1)
        logits=self.fc_layer(logits_2)
        #print(logits.size())
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
        model = CNN().cuda()
    else:
        model = CNN()

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

    #save models
    #method 1:save whole model
    torch.save(obj=model,f="whole_model.pkl")
    #method 2:save parameters
    torch.save(obj=model.state_dict(),f="parameters_model.pkl")


#test
def test(X,y):
    #load model
    #method1:load whole model
    model=torch.load(f="whole_model.pkl")
    #print(model)
    logits=model.forward(input=X)
    prediction = torch.argmax(input=logits, dim=1)
    accuracy = accuracy_score(y_true=y.cpu(), y_pred=prediction.cpu().numpy())
    print("accuracy:",accuracy)



if __name__=="__main__":
    print("Loading Data")
    X_train, y_train, X_valid, y_valid, X_test = load_mnist(path="../data/MNIST/")
    print(X_train.shape)
    X_train=np.reshape(a=X_train,newshape=(-1,1,28,28))
    X_valid = np.reshape(a=X_valid, newshape=(-1, 1, 28, 28))

    #print(X_train[0,0])

    # trans to tensors
    X_train = torch.from_numpy(X_train).cuda()
    y_train = torch.from_numpy(y_train).cuda()
    X_valid = torch.from_numpy(X_valid).cuda()
    y_valid = torch.from_numpy(y_valid).cuda()

    # Training
    print("Training Start!")
    train(X=X_train, y=y_train)

    # Testing
    #print("Test Start!")
    #test(X=X_valid, y=y_valid)

