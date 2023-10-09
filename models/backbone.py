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
    def __init__(self, model, task, input_dim):
        #  Mô hình cải tiến của chúng ta
        self.model = model
        self.multiple_choice_model = MultipleChoice(input_dim=input_dim)
        self.token_classification_model = TokenClassification(input_dim=input_dim)
        self.question_answering_model = QuestionAnswering(input_dim=input_dim)
        
        if task == "multiple_choice":
            self.downsteam = self.multiple_choice_model
        elif task == "token_classification":
            self.downsteam = self.token_classification_model
        elif task == "question_answering":
            self.downsteam = self.question_answering_model
        
        
    def forward(self, input_ids, mask):
        _x = self.model(input_ids, mask)
        _x = self.downsteam(_x)
        return _x