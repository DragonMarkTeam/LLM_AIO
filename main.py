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
from dataloader.DataGenerator import *
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

    train = Trainer(model,
                    loss=loss_fn,
                    optimizer=optimizer,
                    time_limit=time_limit
                    )
    
    train.fit(multi_choice_data_path,
            next_token_predict_data_path,
            batch_size=kwargs.get('batch_size', 32),
            epochs=kwargs.get('epochs', 100),
            checkpoint_directory=kwargs.get('checkpoint_directory', ""),
            context_length=kwargs.get('context_length', 128")
            callbacks=kwargs.get('callbacks', None),
            **kwargs)