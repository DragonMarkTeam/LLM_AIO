import torch
from torch import nn
import collections

class MultipleChoiceHead(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(MultipleChoiceHead, self).__init__()
        self.pooler = torch.nn.Sequential(
            collections.OrderedDict(
                [
                    ("dense", nn.Linear(in_features=input_dim, out_features=input_dim, bias=True)),
                    ("activation", nn.Tanh()),
                ]
            )
        )
        self.dropout =  nn.Dropout(p=0.1, inplace=False)
        self.linear = nn.Linear(in_features=input_dim, out_features=num_labels, bias=True)
        
    def forward(self, input):
        x = self.pooler(input)
        x = self.dropout(input)
        x = self.linear(input)
        return self.decoder(x)
    
    def get_answer(self, input_ids, answer_start_scores, answer_end_scores, tokenizer):
        answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
        answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        return answer