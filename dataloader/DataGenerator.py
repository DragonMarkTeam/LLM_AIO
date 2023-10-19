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


class DataGenerator(Dataset):
    """
    Data Generator
    """

    def __init__(self, dataset_name, tokenizer, split, root_path):
        super(DataGenerator, self).__init__()
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer

        dataset_path = os.path.join(os.path.join(os.path.join(root_path, dataset_name), split), dataset_name+".csv")
        dataset = self.__load__(dataset_path)
        self.dataset = self.__retrieve_columns__(dataset)

    def __load__(self, dataset_path):
        df = pd.read_csv(dataset_path)
        return df
    def __retrieve_columns__(self, df):
        if self.dataset_name == "CulturaX":
            self.columns = ["text"]
            df = Dataset.from_pandas(df)
            tokenized_datasets = df.map(
                self.__tokenize__, batched=True, remove_columns=self.columns
            )
            return tokenized_datasets
        elif self.dataset_name == "ai2_arc_vi":
            self.columns = ["question", "choices", "answerKey"]
            df = Dataset.from_pandas(df)
            tokenized_datasets = df.map(
                self.__tokenize__, batched=True, remove_columns=self.columns
            )
            return tokenized_datasets
        elif self.dataset_name == "arithmetic_vi":
            self.columns = ["context", "completion"]
            df = Dataset.from_pandas(df)
            tokenized_datasets = df.map(
                self.__tokenize__, batched=True, remove_columns=self.columns
            )
        elif self.dataset_name == "grade_12_exams":
            self.columns = ["question", "choices", "answerKey"]
            df = Dataset.from_pandas(df)
            tokenized_datasets = df.map(
                self.__tokenize__, batched=True, remove_columns=self.columns
            )
        elif self.dataset_name == "truthful_qa":
            self.columns = ["question", "mc1_targets"]
            df = Dataset.from_pandas(df)
            tokenized_datasets = df.map(
                self.__tokenize__, batched=True, remove_columns=self.columns
            )
        elif self.dataset_name == "truthful_qa_multi_label":
            self.columns = ["question", "mc2_targets"]
            df = Dataset.from_pandas(df)
            tokenized_datasets = df.map(
                self.__tokenize__, batched=True, remove_columns=self.columns
            )
        return df[self.columns]
    def __tokenize__(batch):
        outputs = tokenizer(
            batch['text'],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
            padding='max_length'
        )
        return outputs[['input_ids', 'token_type_ids', 'length', 'attention_mask']]

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
