from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from accelerate import Accelerator
import evaluate
import torch
import os

def tokenize_function(tokenizer, example):
    return tokenizer(example['sentence1'],example['sentence2'],truncation=True)

def select_column(tokenized_datasets, remove_columns, columns, rename_column):
    tokenized_datasets = tokenized_datasets.remove_columns(remove_columns)
    tokenized_datasets = tokenized_datasets.rename_column(columns, rename_column)
    tokenized_datasets.set_format("torch")
    # tokenized_datasets["train"].column_names
    return tokenized_datasets

def get_dataset(opt, dataset, split):
        
    tokenizer = AutoTokenizer.from_pretrained(opt.pretrain_tokenizer)

    tokenized_datasets = raw_datasets.map(tokenize_function,batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
    tokenized_datasets = select_column(tokenized_datasets, remove_columns, columns, rename_column)
    dataset = DataLoader(tokenized_datasets[split], shuffle=opt.shuffle, batch_size=opt.batch_size, collate_fn=data_collator)
    return dataset

class Trainer(nn.Module):
    def __init__(self, model, optimizer, get_scheduler, opt):
        self.model = model.to(opt.device)
        self.optimizer = optimizer
        self.lr_scheduler = get_scheduler('linear',optimizer =optimizer,num_warmup_steps=0,num_training_steps =opt.num_training_steps)
        self.device = opt.device
        self.accelerator = Accelerator()
    def forward(self, opt):
        metric = evaluate.load('glue','mrpc')
        if self.ckpt != None:
            epoch, self.model, self.optimizer, loss = load_checkpoint(self.ckpt, self.model, self.optimizer)
            if epoch < self.EPOCHS:
                self.train(train_path=self.train_path, valid_path=self.valid_path, cur_epoch=epoch, loss=loss)
                
                torch.save(self.model.state_dict(), os.path.join(self.save_model,"{self.model.__class__.__name__}.pk"))
            else:
                report = self.infer(test_path=self.test_path)
                with open(os.path.join(self.save_report,'{}_continue_report.txt'.format(self.model.__class__.__name__)), 'w') as file:
                    # Write the report content to the file using write()
                    file.write(report)
        else:
            self.train(train_path=self.train_path, valid_path=self.valid_path)
            report = self.infer(test_path=self.test_path)
            with open(os.path.join(self.save_report,'{}_report.txt'.format(self.model.__class__.__name__)), 'w') as file:
                # Write the report content to the file using write()
                file.write(report)
        
    def train_process(self, train_dataloader, valid_dataloader, opt):
        for epoch in tqdm(range(opt.num_epochs)):
            for batch in train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                self.accelerator.backward(loss)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.progress_bar.update(1)
            
            for batch in valid_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                
    def evaluate_process(self, eval_dataloader, metric):
        self.model.eval()
        for batch in eval_dataloader:
            batch = {k:v.to(self.device) for k,v in batch.item()}
            with torch.no_grad():
                outputs = self.model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits,dim= -1)
            metric.add_batch(predictions = predictions,references = batch['labels'])

        metric.compute()
            