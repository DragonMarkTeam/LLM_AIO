"""
DataLoader for the dataset.
TODO: Update this file later
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HuggingfaceDataset
from datasets import load_dataset


class DataGenerator1(Dataset):
    """
    Data Generator
    """

    def __init__(self, data_path, split, tokenizer, inputs_columns, instruction_columns=None, labels_columns=None):
        super().__init__()
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
        return torch.Tensor([self.tokenizer.encode(line)])

    def prompt_state(self):
        return self.have_prompt


class DataGenerator(Dataset):
    """
    Data Generator class. Used for generating data for the model during training.
    """

    def __init__(
        self,
        dataset: HuggingfaceDataset,
        tokenizer,
        batch_size: int,
        features_columns: str,
        labels_columns: str,
        shuffle=True,
        seed=None
    ):
        # Storing Dataset
        self.dataset = dataset
        self.features_columns = features_columns
        self.labels_columns = labels_columns
        # Shuffle dataset
        self.shuffle = shuffle
        self.seed = seed
        # Indexes of datasets
        self.indexes = self.__get_indexes()
        # Tokenizer
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def __get_indexes(self):
        if self.shuffle:
            if self.seed is not None:
                np.random.seed(self.seed)
            return np.random.permutation(range(self.dataset.num_rows))
        return np.arange(self.dataset.num_rows)

    def __tokenize(self, texts):
        return torch.Tensor([self.tokenizer.encode(text) for text in texts])

    def __len__(self):
        return self.dataset.num_rows

    def __iter__(self):
        for i in range(int(self.dataset.num_rows/self.batch_size)):
            indexes = self.indexes[i*self.batch_size:(i+1)*self.batch_size]
            yield self.__tokenize(self.dataset[indexes][self.features_columns])

    def __getitem__(self, index):
        return self.__tokenize(self.dataset[[index]][self.features_columns])

    def split(self, ratio: float):
        """
        Split the dataset into two datasets with the given ratio.
        """
        split_indexes = np.split(
            self.indexes, [int(ratio*self.dataset.num_rows)])
        # Initialize new datasets
        left_dataset = DataGenerator(
            self.dataset, self.tokenizer, self.batch_size, self.features_columns, self.labels_columns, self.shuffle, self.seed)
        right_dataset = DataGenerator(
            self.dataset, self.tokenizer, self.batch_size, self.features_columns, self.labels_columns, self.shuffle, self.seed)
        # Set indexes
        left_dataset.indexes = split_indexes[0]
        right_dataset.indexes = split_indexes[1]
        return left_dataset, right_dataset


class DataAliasGenerator:
    """
    Get DataGenerator by the alias of dataset.
    """

    def __init__(self, dataset_name: str):
        self.dataset = load_dataset(dataset_name)
