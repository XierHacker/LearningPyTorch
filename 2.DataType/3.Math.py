import numpy as np
import torch


a=torch.FloatTensor([[1,2],[3,4]])
b=torch.FloatTensor([[1,2],[3,4]])
#print(a)

def matrix_multiply():
    result=torch.mm(a,b)
    print(result)

def add():
    pass

if __name__=="__main__":
    matrix_multiply()

