from transformers import DataCollatorWithPadding
import torch
from torch import nn
import collections

class TextClassifierHead(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(TextClassifierHead, self).__init__()
        self.transform = torch.nn.Sequential(
            collections.OrderedDict(
                [
                    ("dense", nn.Linear(in_features=input_dim, out_features=input_dim, bias=True)),
                    ("transform_act_fn", nn.GELUActivation()),
                    ("LayerNorm", nn.LayerNorm((768,), eps=1e-12, elementwise_affine=True))
                ]
            )
        )
        self.decoder = nn.Linear(in_features=input_dim, out_features=num_labels, bias=True)
        
    def forward(self, input):
        x = self.transform(input)
        return self.decoder(x)