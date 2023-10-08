import torch
from torch import nn
from models.fine_tuning_model import *
from models.downsteam_task.text_classification import *
from models.downsteam_task.summarization import *
from models.downsteam_task.language_modeling import *
from models.downsteam_task.multiple_choice import *
from models.downsteam_task.token_classification import *
from models.downsteam_task.translation import *
from models.downsteam_task.question_answering import *

class Backbone(nn.Module):
    def __init__(self, model, pretrain_model):
        #  Mô hình cải tiến của chúng ta
        self.model = Fine_tuning_model(
            
        self.text_classification_model = 
        self.summarization_model = 
        self.language_modeling_model = 
        self.multiple_choice_model = 
        self.token_classification_model = 
        self.translation_model = 
        self.question_answering_model = 
        
        
    def forward(self, x):
        