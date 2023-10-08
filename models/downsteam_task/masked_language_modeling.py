from transformers import DataCollatorWithPadding
import torch
from torch import nn

class MaskedLanguageModeling(nn.Module):
    def __init__(self, input_dim, num_choices):
        super(MaskedLanguageModeling, self).__init__()
        self.linear = nn.Linear(input_dim, num_choices)
    def forward(self, input):
        x = self.linear(input)
        return torch.softmax(x, dim=1)