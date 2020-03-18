# Original: https://github.com/pytorch/examples/blob/490243127c02a5ea3348fa4981ecd7e9bcf6144c/time_sequence_prediction/train.py

import torch
import torch.nn as nn

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future):
        outputs = []
        h_t  = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t  = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        # EDIT(momohatt): Add 'dim='
        outputs = torch.stack(outputs, dim=1).squeeze(dim=2)
        return outputs


# Example input
def gen_Sequence_test():
    model = Sequence()
    model.double()
    x = torch.rand(3, 4)
    future = 0
    return model, (x, future)
