import torch
from transformers import AutoModel, AutoTokenizer
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

class CustomDataset(Dataset):
    def __init__(self, data_path, split, tokenizer, inputs_columns, instruction_columns=None, labels_columns=None):
        super(CustomDataset, self).__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        
        wiki = load_dataset("arrow", data_files={split: self.data_path})
        self.inputs = wiki[split][inputs_columns]
        if instruction_columns == None:
            self.have_prompt = False
            self.prompts = [''] * len(self.inputs)
        else:
            self.have_prompt = True
            self.prompts = wiki[split][instruction_columns]
            
            
        self.labels = wiki[split][labels_columns]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.__tokenizer_string__(self.inputs[idx]), self.__tokenizer_string__(self.prompts[idx]), self.__tokenizer_string__(self.labels[idx])
    def __tokenizer_string__(self, line):
        return torch.tensor([self.tokenizer.encode(line)])
    def prompt_state(self):
        return self.have_prompt