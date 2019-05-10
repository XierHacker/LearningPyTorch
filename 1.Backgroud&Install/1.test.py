import torch
import numpy


print("version:",torch.__version__)

if torch.cuda.is_available():
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_capability())
    print(torch.cuda.get_device_name())
    print(torch.cuda.get_device_properties(device=torch.device("cuda:0")))
