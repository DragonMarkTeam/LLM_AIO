import torch
from torch import nn
import collections

class NextSentencePredictionHead(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(NextSentencePredictionHead, self).__init__()
        self.pooler = torch.nn.Sequential(
            collections.OrderedDict(
                [
                    ("dense", nn.Linear(in_features=input_dim, out_features=input_dim, bias=True)),
                    ("activation", nn.Tanh()),
                ]
            )
        )
        self.NSPHead = torch.nn.Sequential(
            collections.OrderedDict(
                [
                    ("seq_relationship", nn.Linear(in_features=input_dim, out_features=2, bias=True))
                ]
            )
        )
        
    def forward(self, input):
        x = self.pooler(input)
        x = self.NSPHead(x)
        return x