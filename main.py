import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM


# ======================================================================
# ======================================================================
# ======================================================================
from models.fine_tuning_model import *
from models.backbone import *
from train.config import *
from train.train import *
from dataloader.dataset import *
# ======================================================================
# ======================================================================
# ======================================================================

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

if __name__ == '__main__':
    kwargs = train_config()
    
    fine_tuning_model = SharedModel(kwargs.get('pretrain_model', 'llama-2'))
    model = Backbone(fine_tuning_model, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=kwargs.get('learning_rate', 1e-5))
    
    train_generator = DataLoader(DataGenerator(data_path=kwargs.get('learning_rate', 1e-5),
                                               split="train",
                                               tokenizer=tokenizer,
                                               inputs_columns=kwargs.get('learning_rate', 1e-5),
                                               instruction_columns=kwargs.get('learning_rate', 1e-5),
                                               labels_columns=kwargs.get('learning_rate', 1e-5)))
    valid_generator = DataLoader(DataGenerator(data_path=kwargs.get('learning_rate', 1e-5),
                                               split="test",
                                               tokenizer=tokenizer,
                                               inputs_columns=kwargs.get('learning_rate', 1e-5),
                                               instruction_columns=kwargs.get('learning_rate', 1e-5),
                                               labels_columns=kwargs.get('learning_rate', 1e-5)))
    
    train = Trainer(train_generator,
        valid_generator,
        batch_size=kwargs.get('batch_size', 32),
        epochs=kwargs.get('epochs', 100),
        checkpoint_directory=kwargs.get('checkpoint_directory', ""),
        callbacks=kwargs.get('callbacks', None),
        **kwargs)
    
    train()