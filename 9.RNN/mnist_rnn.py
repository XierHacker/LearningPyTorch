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
NUM_HIDDEN=300
NUM_OUT=10

class LSTM_MNIST(torch.nn.Module):
    def __init__(self):
        super(LSTM_MNIST, self).__init__()
        self.lstm=torch.nn.LSTM(input_size=28,hidden_size=NUM_HIDDEN,num_layers=3,batch_first=True)
        self.fc_layer=torch.nn.Linear(in_features=NUM_HIDDEN,out_features=NUM_OUT)

    def forward(self, input):
        output,_=self.lstm(input)
        output=output[:,-1,:]       #shape[batch_size,num_hidden]
        logits=self.fc_layer(output)
        return logits


'''
train
'''
#parameters
MAX_EPOCHS=5
BATCH_SIZE=128
LEARNING_RATE=0.0002

def train(X,y):
    #model=MLP()
    #use model
    if torch.cuda.is_available():
        model = LSTM_MNIST().cuda()
    else:
        model = LSTM_MNIST()

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
    X_train=np.reshape(a=X_train,newshape=(-1,28,28))
    X_valid = np.reshape(a=X_valid, newshape=(-1, 28, 28))

    #print(X_train[0,0])

    # trans to tensors
    X_train = torch.from_numpy(X_train).cuda()
    y_train = torch.from_numpy(y_train).cuda()
    X_valid = torch.from_numpy(X_valid).cuda()
    y_valid = torch.from_numpy(y_valid).cuda()

    # Training
    #print("Training Start!")
    #train(X=X_train, y=y_train)

    # Testing
    print("Test Start!")
    test(X=X_valid, y=y_valid)
