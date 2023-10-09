import torch
from torch import nn
import collections

class TokenClassification(nn.Module):
    def __init__(self, input_dim):
        super(TokenClassification, self).__init__()
        self.dropout =  nn.Dropout(p=0.1, inplace=False)
        self.linear = nn.Linear(in_features=input_dim, out_features=2, bias=True)
        
    def forward(self, input):
        x = self.dropout(input)
        x = self.linear(input)
        return x