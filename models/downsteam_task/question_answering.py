import torch
from torch import nn
import collections

class QuestionAnsweringHead(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(QuestionAnsweringHead, self).__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=num_labels, bias=True)
        
    def forward(self, input):
        x = self.linear(input)
        return x