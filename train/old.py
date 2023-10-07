from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from accelerate import Accelerator
import torch
import os
from torch.cuda.amp import autocast, GradScaler
from datasets import load_from_disk
import copy
# def tokenize_function(tokenizer, example):
#     return tokenizer(example['sentence1'],example['sentence2'],truncation=True)

# def get_dataset(opt, dataset, split):
        
#     tokenizer = AutoTokenizer.from_pretrained(opt.pretrain_tokenizer)

#     tokenized_datasets = raw_datasets.map(tokenize_function,batched=True)
#     data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#     dataset = DataLoader(tokenized_datasets[split], shuffle=opt.shuffle, batch_size=opt.batch_size, collate_fn=data_collator)
#     return dataset

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
    def train(self, train_path, valid_path, cur_epoch=0, loss=1_000_000):
        training_loader = DataLoader(load_from_disk(train_path)["train"], 
                                     batch_size=self.batch_size,
                                     num_workers=self.num_workers,
                                     pin_memory=self.pin_memory,
                                     shuffle=self.shuffle)
        validation_loader = DataLoader(load_from_disk(valid_path)["validation"], 
                                       batch_size=self.batch_size,
                                       num_workers=self.num_workers,
                                       pin_memory=self.pin_memory,
                                       shuffle=self.shuffle)
        
        self.best_vloss = loss
        
        # Define your scaler for loss scaling
        scaler = GradScaler()
    
        for epoch in range(cur_epoch, self.EPOCHS, 1):
            # Training
            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.training_step(training_loader, epoch, scaler)
            
            # Set the model to evaluation mode, disabling dropout and using population 
            # statistics for batch normalization.
            self.model.eval()
            avg_vloss = self.validation_step(validation_loader, epoch)
            
            print('LOSS train {}, valid {}'.format(avg_loss, avg_vloss))
            
            # Track best performance, and save the model's state
            if avg_vloss < self.best_vloss:
                self.best_vloss = avg_vloss
                self.best_weights = copy.deepcopy(self.model.state_dict())
                save_checkpoint(PATH=self.save_checkpoint, EPOCH=epoch, model=self.model, optimizer=self.optimizer, LOSS=avg_vloss)
                # model_path = self.save_path+'model_{}_{}'.format(timestamp, epoch)
                # torch.save(self.model.state_dict(), model_path)
                
            if not torch.cuda.is_available():
                save_checkpoint(PATH=self.save_checkpoint, EPOCH=epoch, model=self.model, optimizer=self.optimizer, LOSS=avg_vloss)
        
        # restore model and return best accuracy
        self.model.load_state_dict(self.best_weights)
        return self.best_vloss
    
    def infer(self, test_path):
        testing_loader = DataLoader(load_from_disk(test_path)["test"],
                                    batch_size=self.batch_size,
                                    num_workers=self.num_workers,
                                    pin_memory=self.pin_memory,
                                    shuffle=self.shuffle)
        return self.testing_step(testing_loader)
        
    def training_step(self, training_loader, epoch_index, scaler):
        for i, data in tqdm(enumerate(training_loader), desc=f'Epoch valid {epoch_index}/{self.EPOCHS}', initial=0, dynamic_ncols=True):
            # Every data instance is an input + label pair
            inputs, extra_inputs, labels = data
                    
            inputs, extra_inputs, labels = inputs.to(self.device), extra_inputs.to(self.device), labels.to(self.device)
                    
            # Zero your gradients for every batch!
            self.optimizer.zero_grad(set_to_none=True)
            # Enable autocasting for forward pass
            with autocast():
                y_preds = self.model(inputs, extra_inputs)
                loss = self.loss_fn(y_preds, labels)
                        
            # Perform backward pass and optimization using scaler
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
                
    def validation_step(self, validation_loader, epoch_index):
        self.model.eval()
        running_vloss = 0.0
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in tqdm(enumerate(validation_loader), desc=f'Epoch valid {epoch_index}/{self.EPOCHS}', initial=0, dynamic_ncols=True):
                inputs, extra_inputs, labels = vdata
                    
                inputs, extra_inputs, labels = inputs.to(self.device), extra_inputs.to(self.device), labels.to(self.device)
                            
                with autocast():
                    y_preds = self.model(inputs, extra_inputs)
                    vloss = self.loss_fn(y_preds, labels)
                    running_vloss += vloss
                                    
                avg_vloss = running_vloss / (i + 1)
            return avg_vloss

def save_checkpoint(PATH, EPOCH, model, optimizer, LOSS):
    path = PATH+"/"+str(EPOCH)+".pk"
    torch.save({
                'epoch': EPOCH,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': LOSS,
                }, path)
    
def load_checkpoint(PATH, model, optimizer):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return epoch, model, optimizer, loss