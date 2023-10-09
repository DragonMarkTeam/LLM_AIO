import torch
from torch import nn
import collections

class QuestionAnswering(nn.Module):
    def __init__(self, input_dim):
        super(QuestionAnswering, self).__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=2, bias=True)
        
    def forward(self, input):
        x = self.linear(input)
        return x