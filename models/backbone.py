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
from models.downsteam_task.masked_language_modeling import *

class Backbone(nn.Module):
    def __init__(self, model, **kwargs):
        #  Mô hình cải tiến của chúng ta
        self.model = model
        self.shared_model_output_size = kwargs.get("shared_model_output_size")
        self.num_labels =  kwargs.get("num_labels")
        self.num_choices =  kwargs.get("num_choices") # 1
        self.num_text_classification =  kwargs.get("num_text_classification") # 2
        self.Q_A_position =  kwargs.get("Q_A_position") # 2
        
        self.multiple_choice_model = MultipleChoiceHead(input_dim=self.shared_model_output_size, num_labels=self.num_choices)
        self.token_classification_model = TokenClassificationHead(input_dim=self.shared_model_output_size, num_labels=self.num_text_classification)
        self.question_answering_model = QuestionAnsweringHead(input_dim=self.shared_model_output_size, num_labels=self.Q_A_position)
        self.masked_language_modeling = MaskedLanguageModelingHead(input_dim=self.shared_model_output_size, num_labels=self.num_labels)
        
        if kwargs.get("task") == "multiple_choice":
            self.downsteam = self.multiple_choice_model
        elif kwargs.get("task") == "token_classification":
            self.downsteam = self.token_classification_model
        elif kwargs.get("task") == "question_answering":
            self.downsteam = self.question_answering_model
        elif kwargs.get("task") == "masked_language_modeling":
            self.downsteam = self.question_answering_model
        
        
    def forward(self, input_ids, mask):
        _x = self.model(input_ids, mask)
        _x = self.downsteam(_x)
        return _x