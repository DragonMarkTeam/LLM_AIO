# VLSP 2023. (AIO)

This file is used as Training class for Large Language Models.

## Document

Training Pretrain model with Trainer handler.

```py
from train import Trainer
from models.pretrains import GPT2

pretrain_model = GPT2()
trainer = Trainer(model=pretrain_model)
trainer.fit(
    train_generator,
    valid_generator,
    batch_size=256,
    epochs=40,
    checkpoint_directory="/path/to/folder"
)
```

Create Generator

```py
from dataloader import DataGenerator
# Module from huggingface
from datasets import Dataset

ds = Dataset.from_file("example.arrow")

data_gen = DataGenerator(ds, shuffle=True, batch_size=256)
train_generator, valid_generator = data_gen.split(ratio=0.2, seed=42)

# Get items. Index slicing based on batch size.
# Output format (batch_size, context_length)
train_generator[0]
```

Advance configuation of creating trainer

```py
# Module import from pytorch
from torch.optim import Adam()
from torch.nn import MeanSquareError()

# Set custom optimizer
pretrain_model = GPT2()
trainer.fit(
    ...
    optimizer=Adam(pretrain_model.parameters(), lr=1e-5)
)

# Set custom loss
trainer.fit(
    ...
    loss=MeanSquareError()
)

# Set timelimit for training. In case someone wanted to training for test.
# Measured on second.
trainer.fit(
    ...
    time_limit=200000
)

# Add middleware for preprocessing predict output before compute loss.
def handle_processing(trainer, preds):
    # trainer: The Trainer class itself. Used for accessed arguments.
    # preds: Predicted value.
    ...
trainer.fit(
    ...
    middleware_preds=handle_processing
)
```
