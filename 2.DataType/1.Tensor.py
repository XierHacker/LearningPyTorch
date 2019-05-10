import numpy as np
import torch


def CreateTensor():
    '''
    创建Tensor的测试
    :return:
    '''
    # create Tensor
    # torch.Tensor(sequence)
    print("Create Tensor Test:")
    tensor_a=torch.tensor(data=[1, 2, 3, 4, 5],dtype=torch.float32,requires_grad=False)
    print("tensor_a:\n",tensor_a)
    print("size of tensor_a:", tensor_a.size())
    print("shape of tensor_a:", tensor_a.shape)
    print("first dimension of tensor_a:", tensor_a.shape[0])

    # torch.Tensor(size)
    tensor_b = torch.zeros(size=(3,3),dtype=torch.int32,requires_grad=False)
    print("tensor_b:\n",tensor_b)
    print("size of tensor_b:", tensor_b.size())


def ndarray2tensor():
    '''
    tensor和ndarray互转案例
    :return:
    '''
    print("ndarray2tensor test:")
    #array to tensor
    a=np.array([1, 2, 3])
    t = torch.from_numpy(a)
    print("tensor from a:\n",t)
    a[0]=100        #改变a同时也会影响到其对应的t
    print("tensor after modified a:\n",t)

    #tensor2array
    print("t.numpy():\n",t.numpy())

    print("t.device:\n",t.device)


def DeviceTest():
    '''
    CPU和GPU管理
    :return:
    '''
    if torch.cuda.is_available():
        x=torch.zeros(size=(3,3))
        print("x:\n",x)
        device = torch.device("cuda")
        y = torch.ones_like(x, device=device)
        x = x.to(device)
        z = x + y
        print("z:\n",z)
        # print("z.data:\n",z.numpy())    错误，GPU上面不能够得到numpy()

        # #要是想转为numpy，需要先从GPU先移动到CPU
        z=z.to(device=torch.device("cpu"),dtype=torch.float32)
        print(z.numpy())




if __name__=="__main__":
    CreateTensor()
    print("\n")
    ndarray2tensor()
    print("\n")
    DeviceTest()
