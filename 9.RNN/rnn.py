import numpy as np
import torch

class SimpleRNN(torch.nn.Module):
    def __init__(self):
        super(SimpleRNN, self).__init__()
        self.rnn=torch.nn.RNN(input_size=28,hidden_size=10,num_layers=2,batch_first=True)
        print("shape of layer1:")
        print("rnn.weight_ih_l0: ",self.rnn.weight_ih_l0.size())
        print("rnn.bias_ih_l0: ",self.rnn.bias_ih_l0.size())
        print("rnn.weight_hh_l0: ",self.rnn.weight_hh_l0.size())
        print("rnn.bias_hh_l0: ", self.rnn.bias_hh_l0.size())


        print("shape of layer2:")
        print("rnn.weight_ih_l1: ",self.rnn.weight_ih_l1.size())
        print("rnn.bias_ih_l1: ",self.rnn.bias_ih_l1.size())
        print("rnn.weight_hh_l1: ",self.rnn.weight_hh_l1.size())
        print("rnn.bias_hh_l1: ", self.rnn.bias_hh_l1.size())

        #print(rnn.weight_ih_l2.size())
        #print(rnn.bias_ih_l2.size())


    def forward(self, input,hidden):
        output,h=self.rnn(input,hidden)
        return output,h



if __name__=="__main__":
    simple_rnn=SimpleRNN()
    print("\n input test:")
    randon_input=torch.randn(40,30,28)  #shape(batch, seq, feature)
    random_h=torch.randn(2,40,10)       #shape:(num_layers * num_directions,batch,hidden_size)
    output,h=simple_rnn(randon_input,random_h)

    print("output",output.size())
    print("h",h.size())

