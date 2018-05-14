'''
    这是pytorch提供的模板，这个文件是伪代码，不能够执行，请仿照这个模板来实现自己的模型
'''
import torch

class net_name(torch.nn.Module):
    def __init__(self,other_argument):
        super(net_name,self).__init__()
        self.conv1=torch.nn.Conv2d(.......)
        #add network main components below

    def forward(self, input):
        logit=self.conv1(input)
        return logit

