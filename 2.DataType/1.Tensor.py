import numpy as np
import torch

#create Tensor examples
def CreateTensor():
    # create Tensor
    # torch.Tensor(sequence)
    print("Create Tensor Test:")
    tensor_a=torch.tensor(data=[1, 2, 3, 4, 5],dtype=torch.float32,requires_grad=False)
    print("tensor_a:",tensor_a)
    print("size of tensor_a:", tensor_a.size())
    print("size of tensor_a:", tensor_a.shape)
    print("first dimension of tensor_a:", tensor_a.shape[0])

    # torch.Tensor(size)
    tensor_b = torch.zeros(size=(3,3),dtype=torch.int32,requires_grad=False)
    print("tensor_b:",tensor_b)
    print("shape of tensor_b:", tensor_b.size())


def ndarray2tensor():
    print("ndarray2tensor test:")
    #array to tensor
    a=np.array([1, 2, 3])
    t = torch.from_numpy(a)
    print(t)
    a[0]=100
    print(t)

    #tensor2array
    print(t.numpy())


if __name__=="__main__":
    CreateTensor()
    ndarray2tensor()