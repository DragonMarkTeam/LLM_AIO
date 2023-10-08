import torch
from torch import nn
import os
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.data import DataLoader
import copy
import json
import datetime

from ..dataloader.dataloader import CustomDataset

class Trainer(nn.Module):
    def __init__(self, 
                 model, 
                 optimizer, 
                 tokenizer, 
                 prompt,
                 checkpoint_path, 
                 model_path, 
                 state_dataset_path, 
                 num_epochs, 
                 train_dataset_path=None, 
                 valid_dataset_path=None, 
                 choosen_columns=None, 
                 tan_suat_save_chechpoint=1, 
                 batch_size=32, 
                 shuffle=False, 
                 num_workers=0, 
                 collate_fn=None, 
                 pin_memory=False, 
                 drop_last=True, 
                 timeout=0, 
                 limit_time_train=10000000,
                 pin_memory_device="", 
                 device="cuda"):
        super(Trainer, self).__init__()
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.num_epochs = num_epochs
        self.train_dataset = None
        self.valid_dataset = None
        self.tan_suat_save_chechpoint = tan_suat_save_chechpoint
        self.choosen_columns = choosen_columns
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.pin_memory_device = pin_memory_device
        
        # Thời gian giới hạn cho việc train, nếu vượt qua mốc này, quá trình train được xem là đã bị break
        self.limit_time_train = limit_time_train
        
        # Load dataset path
        self.train_dataset_path = train_dataset_path
        self.valid_dataset_path = valid_dataset_path
        
        
        # Dataset state
        # Xác nhận thời gian gần nhất mà dataset được load
        self.time_load = self.get_time_now()
        
        # Model state
        # Trọng số mô hình tốt nhất
        self.best_weights = None
        # Xác nhận độ chính xác tốt nhất của mô hình
        self.best_accuracy = -1
        # Xác nhận giá trị loss tương ứng
        self.best_vloss = 1000000
        # Xác nhận epoch mà mô hình đạt đô chính xác tốt nhất
        self.correspond_epoch = 0
        # Xác nhận mốc thời gain mà mô hình đạt đô chính xác tốt nhất
        self.correspond_timestamp = self.time_load
        
        # Trường hợp bị break giữa chừng
        self.cur_epoch = 0
    
        # Save path
        self.checkpoint_path = checkpoint_path
        self.model_path = model_path
        self.state_dataset_path = state_dataset_path 

        
        
        if self.train_dataset_path != None:
            self.train_dataset = DataLoader(dataset=CustomDataset(self.train_dataset_path, 
                                                                  "train", 
                                                                  self.tokenizer, 
                                                                  inputs_columns=self.choosen_columns[0], 
                                                                  instruction_columns=self.choosen_columns[1], 
                                                                  labels_columns=self.choosen_columns[2]),
                                            batch_size=self.batch_size,
                                            shuffle=self.shuffle,
                                            num_workers=self.num_workers,
                                            collate_fn=self.collate_fn,
                                            pin_memory=self.pin_memory,
                                            drop_last=self.drop_last,
                                            timeout=self.timeout,
                                            pin_memory_device=self.pin_memory_device)
                                            
                                            
        if self.valid_dataset_path != None:
            self.valid_dataset = DataLoader(dataset=CustomDataset(self.valid_dataset_path, 
                                                                  "valid", 
                                                                  self.tokenizer, 
                                                                  inputs_columns=self.choosen_columns[0], 
                                                                  instruction_columns=self.choosen_columns[1], 
                                                                  labels_columns=self.choosen_columns[2]),
                                            batch_size=self.batch_size,
                                            shuffle=self.shuffle,
                                            num_workers=self.num_workers,
                                            collate_fn=self.collate_fn,
                                            pin_memory=self.pin_memory,
                                            drop_last=self.drop_last,
                                            timeout=self.timeout,
                                            pin_memory_device=self.pin_memory_device)
        
    def check_break(self, is_training, trained, time_load):
        if is_training == True and trained == False:
            if time_load - self.get_time_now() > self.limit_time_train:
                print("Training process was break.")
                return True
        return False
            
    def check_being_trained(self, is_training, trained, time_load):
        if is_training == True and trained == False:
            if time_load - self.get_time_now() < self.limit_time_train:
                print("Another account is training this dataset.")
                return True
        return False
    
    def get_time_now(self):
        return datetime.datetime.now()
    def train(self):
        # ======================================================================
        # Step by step training
        # 1 - Load_dataset_state: 
        # 1.1 - Kiểm tra có account nào đang train không. Nếu không thì chuyển sang bước 1.2. Nếu có thì break cả file train.py này để chuyển sang dataset mới.
        # 1.2 - Kiểm tra xem giai đoạn train trên dataset đó có bị break không. Nếu có chuyển sang bước 1.2.1. Nếu không thì qua bước 2.
        # 1.2.1 - Load các thông số (break_correspond_model, break_correspond_optimizer, break_best_accuracy, break_correspond_loss, break_correspond_epoch, break_correspond_timestamp) và lưu vào các biến trong trainer class. Và bắt đầu bước 2.
        # 2 - Chạy hàm train. Train và validate xong thì chuyển sang bước 3.
        # 3 - Lưu kết quả
        # 3.1 - Mở file lưu mô hình cũ. Nếu có cải tiến về độ chính xác thì lưu lại.
        # 3.2 - Mở file lưu trạng thái dataset. Nếu có cải tiến thì lưu các giá trị lại. Chú ý, với từng mô hình thì sẽ có file trạng thái riêng cho từng dataset.
        is_training, trained, time_load, break_correspond_model, break_correspond_optimizer, break_best_accuracy, break_correspond_loss, break_correspond_epoch, break_correspond_timestamp = self.load_state_dataset()
        if self.check_being_trained(is_training, trained, time_load)==False:
            return False
        else:
            if self.check_break(is_training, trained, time_load)==False:
                self.load_state_dataset()
        
        self.train_process()
        
        self.save_model()
        self.save_dataset_state()
        
    def train_process(self):
        # Define your scaler for loss scaling
        scaler = GradScaler()
    
        for epoch in range(self.correspond_epoch, self.num_epochs, 1):
            # Training
            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.training_step(scaler)
            
            # Set the model to evaluation mode, disabling dropout and using population 
            # statistics for batch normalization.
            self.model.eval()
            avg_vloss = self.validation_step()
            
            print('LOSS train {}, valid {}'.format(avg_loss, avg_vloss))
            
            # Track best performance, and save the model's state
            if avg_vloss < self.best_vloss:
                self.best_vloss = avg_vloss
                self.best_weights = copy.deepcopy(self.model.state_dict())
                self.save_checkpoint(epoch=epoch)
                
            if not torch.cuda.is_available():
                self.save_checkpoint(epoch=epoch)
        
        # restore model and return best accuracy
        self.model.load_state_dict(self.best_weights)
        return self.best_vloss
    def training_step(self, scaler):
        for data in tqdm(self.train_dataset, desc=f'Epoch train {self.cur_epoch}/{self.num_epochs}', initial=0, dynamic_ncols=True):
            input_ids, attention_mask, token_type_ids, labels = data
            input_ids, attention_mask, token_type_ids, labels = input_ids.to(self.device), attention_mask.to(self.device), token_type_ids.to(self.device), labels.to(self.device)
                    
            # Zero your gradients for every batch!
            self.optimizer.zero_grad(set_to_none=True)
            # Enable autocasting for forward pass
            with autocast():
                y_preds = self.model(inputs, prompts)
                loss = self.loss_fn(y_preds, labels)
                        
            # Perform backward pass and optimization using scaler
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
    
    def validation_step(self):
        self.model.eval()
        running_vloss = 0.0
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in tqdm(enumerate(self.valid_dataset), desc=f'Epoch valid {self.cur_epoch}/{self.num_epochs}', initial=0, dynamic_ncols=True):
                inputs, prompts, labels = vdata
                    
                inputs, prompts, labels = inputs.to(self.device), prompts.to(self.device), labels.to(self.device)
                            
                with autocast():
                    y_preds = self.model(inputs, prompts)
                    vloss = self.loss_fn(y_preds, labels)
                    running_vloss += vloss
                                    
                avg_vloss = running_vloss / (i + 1)
            return avg_vloss
        
    def load_model(self):
        # Opening JSON file
        with open(self.model_path, 'r') as openfile:
            # Reading from json file
            json_object = json.load(openfile)
            return json_object["model"], json_object["optimizer"], json_object["best_accuracy"], json_object["correspond_loss"], json_object["correspond_epoch"], json_object["correspond_dataset"], json_object["correspond_timestamp"]
    
    def save_checkpoint(self, epoch):
        # Data to be written
        dictionary = {
            "model": self.model.__class__.__name__ ,
            "accuracy": self.best_accuracy,
            "loss": self.best_vloss,
            "epoch": epoch,
            "timestamp": self.get_time_now()
        }
        # Serializing json
        json_object = json.dumps(dictionary, indent=4)
        
        # Writing to sample.json
        with open(self.checkpoint_path + "_" +str(self.model.__class__.__name__) + "_" + str(self.get_time_now()) + "_" + str(self.cur_epoch) + ".json", "w") as outfile:
            outfile.write(json_object)
            
    def save_model(self):
        # Đầu tiên là load lên cái thông số cũ.
        old_model, old_optimizer, old_best_accuracy, old_correspond_loss, old_correspond_epoch, old_correspond_dataset, old_correspond_timestamp = self.load_model()
        # Kiểm tra nếu độ chính xác của mô hình cải thiện thì mới lưu lại
        if old_best_accuracy < self.best_accuracy:
            dictionary = {
            "model": self.model.__class__.__name__ ,
            "optimizer": self.optimizer,
            "accuracy": self.best_accuracy,
            "loss": self.best_vloss,
            "epoch": self.correspond_epoch,
            "dataset": self.valid_dataset_path,
            "timestamp": self.get_time_now()
            }
            # Serializing json
            json_object = json.dumps(dictionary, indent=4)
            
            # Writing to sample.json
            with open(self.model_path, "w") as outfile:
                outfile.write(json_object)
                
    def load_state_dataset(self):
        is_training, trained, time_load, break_correspond_model, break_correspond_optimizer, break_best_accuracy, break_best_vloss, break_correspond_epoch, break_correspond_timestamp = self.load_state_dataset()
        # ========================================================================
        # Có một cái hàm assert, nhưng mà quên cú pháp rồi
        # ========================================================================

        self.cur_epoch = break_correspond_epoch
        self.model = break_correspond_model
        self.best_accuracy = break_best_accuracy
        self.best_vloss = break_best_vloss
        self.optimizer = break_correspond_optimizer
        self.correspond_timestamp = break_correspond_timestamp
    def change_state_dataset(self, is_training, trained):
        with open(self.model_path, 'r') as openfile:
            # Reading from json file
            json_object = json.load(openfile)
            # Xác nhận là dataset này đã được load để train
            json_object["is_training"] = is_training
            # Xác nhận là model này chưa train xong dataset này.
            json_object["trained"] = trained
            # Xác nhận thời gian gần nhất mà dataset được load
            json_object["timestamp"] = self.get_time_now()
            
        # Writing to sample.json
        with open(self.model_path, "w") as outfile:
            outfile.write(json_object)