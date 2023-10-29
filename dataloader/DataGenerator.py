"""
DataLoader for the dataset.
TODO: Update this file later
"""
import torch
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import os
import csv
import pandas as pd


class DataGenerator(torch.nn.Module):
    def __init__(self, list_path, type, tokenizer, context_length):
        super(DataGenerator, self).__init__()
        self.tokenizer = tokenizer
        self.context_length = context_length
        if type == "multiple_choice":
        self.dataset = Dataset.from_dict(self.multiple_choice_tokenize_function(self.multiple_choice_load_dataframe(list_path)))
        elif type == "next_token_predict":
        self.dataset = Dataset.from_dict(self.next_token_tokenize_function(self.next_token_load_dataframe(list_path)))
    def train_test_split(self, test_size):
        return self.dataset.train_test_split(test_size=test_size).values()
    def multiple_choice_load_dataframe(self, multiple_choice_list_path):
        for i in range(len(multiple_choice_list_path)):
        temp_dataset = pd.read_csv(multiple_choice_list_path[i])
        if i == 0:
            dataset = temp_dataset
        else:
            dataset = pd.concat([dataset, temp_dataset], axis=1)
        return Dataset.from_pandas(dataset)
    def next_token_load_dataframe(self, next_token_list_path):
        for i in range(len(next_token_list_path)):
        temp_dataset = pd.read_csv(next_token_list_path[i])
        if i == 0:
            dataset = temp_dataset
        else:
            dataset = pd.concat([dataset, temp_dataset], axis=1)
        return Dataset.from_pandas(dataset)
    def multiple_choice_tokenize_function(self, batch):
        prompt = self.tokenizer(batch['prompt'],
                                padding='max_length',
                                truncation=True,
                                max_length=self.context_length)
        label = self.tokenizer(batch['label'],
                                padding='max_length',
                                truncation=True,
                                max_length=self.context_length)
        return {'input_ids': prompt['input_ids'],
                'attention_mask_prompt': prompt['attention_mask'],
                'label': label['input_ids'],
                'attention_mask_label': label['attention_mask']}
    def next_token_tokenize_function(self, batch):
        prompt = self.tokenizer(batch['text'],
                                padding='max_length',
                                truncation=True,
                                max_length=self.context_length)
        return {'input_ids': prompt['input_ids'],
                'attention_mask_prompt': prompt['attention_mask'],
                'label': prompt['input_ids'],
                'attention_mask_label': prompt['attention_mask']}

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)