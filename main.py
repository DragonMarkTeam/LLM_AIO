from transformers import AdamW
from transformers import get_scheduler

optimizer = AdamW(model.parameters(),lr=5e-5)
lr_scheduler = get_scheduler('linear',optimizer =optimizer,num_warmup_steps=0,num_training_steps =num_training_steps)