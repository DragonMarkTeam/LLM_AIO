from transformers import DataCollatorWithPadding
import torch
from torch import nn

class TextClassifier(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(TextClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_labels)
    def forward(self, input):
        x = self.linear(input)
        return torch.softmax(x, dim=1)