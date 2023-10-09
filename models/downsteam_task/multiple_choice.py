import torch
from torch import nn
import collections

class MultipleChoice(nn.Module):
    def __init__(self, input_dim):
        super(MultipleChoice, self).__init__()
        self.pooler = torch.nn.Sequential(
            collections.OrderedDict(
                [
                    ("dense", nn.Linear(in_features=input_dim, out_features=input_dim, bias=True)),
                    ("activation", nn.Tanh()),
                ]
            )
        )
        self.dropout =  nn.Dropout(p=0.1, inplace=False)
        self.linear = nn.Linear(in_features=input_dim, out_features=1, bias=True)
        
    def forward(self, input):
        x = self.pooler(input)
        x = self.dropout(input)
        x = self.linear(input)
        return self.decoder(x)