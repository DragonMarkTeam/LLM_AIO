"""
Training module. This file contains the Trainer class which used to train the model.
"""

import os
import time
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.optim import Optimizer, AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from .callbacks import BaseCallback
from ..utils.cast_device import cast_to_device


CHECKPOINT_EXTENSION = '.pt'


class Trainer:
    """
    Trainer Wrapper modules. Every models will be called inside here.

    Example:
    >>> from train import Trainer
    >>> from train.callbacks import EarlyStopping
    >>> from models import GPT2

    >>> model = GPT2()
    >>> trainer = Trainer(model)
    >>> trainer.fit(
    >>>     train_generator,
    >>>     valid_generator,
    >>>     batch_size=256,
    >>>     epochs=40,
    >>>     checkpoint_directory='/checkpoints',
    >>>     callbacks=[EarlyStopping()]
    >>> )

    Args:
        train_generator (torch.utils.data.DataLoader): Training data generator.
        valid_generator (torch.utils.data.DataLoader): Validation data generator.
        model (torch.nn.Module): Model to be trained.
        loss (torch.nn.Module): Loss function to be used.
        optimizer (torch.optim.Optimizer): Optimizer to be used.
        batch_size (int): Batch size.
        epochs (int): Epochs.
        checkpoint_directory (str): Directory to save model checkpoints.
        callbacks (list): List of callbacks to be used.
        time_limit (int): Limit time for training process. If None, it will be ignored.
        device (str): Device to be used. Default: 'cuda' if available, otherwise 'cpu'.

    Methods:
        fit: Fit model with given data generator.
        load_weights: Load model's weights from checkpoint.

    Keyword Args:
        learning_rate (float): Learning rate. Default: 1e-5.
    """

    def __init__(
        self,
        model: nn.Module,
        loss=None,
        optimizer: Optimizer = None,
        time_limit: int = None,
        **kwargs
    ):
        # --- Data inputs args ---
        # The train_generator and valid_generator will be set in fit method
        self.train_generator = None
        self.valid_generator = None

        # --- Model configs args ---
        # The model variable will be set in fit method
        self.model = model

        # The loss is set to nn.CrossEntropyLoss() by default
        self.loss = loss if loss else nn.CrossEntropyLoss()

        # The optimizer is set to AdamW by default
        self.optimizer = optimizer if optimizer else AdamW(
            self.model.parameters(), lr=kwargs.get('learning_rate', 1e-5)
        )

        # --- Training args ---
        self.batch_size = None
        self.epochs = None

        # --- Checkpoint args ---
        self.checkpoint_directory = None

        # --- Callbacks args ---
        # The callbacks variable will be set in fit method
        self.callbacks: list[BaseCallback] = []

        # Define histories which includes train_loss and valid_loss
        self.histories = {
            'train_loss': [],
            'valid_loss': []
        }

        # --- Utils options args ---
        # If the training time reach the time_limit, the training process will be stopped
        # By default, the time_limit is set to None, which unlimit the training time
        self.time_limit = time_limit
        self.training_start_time = 0

        # Define device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Define best valid loss for tracking best model
        self.best_valid_loss = 1e6

        # Define some default kwargs
        self.next_checkpoint_name_id = 1
        self.middleware_preds = None
        self.kwargs = kwargs

    def __before_training(self):
        """
        (private) Before training phase.
        --------------------------------
        This function will be called before `__training()`.
        By default, this function will call `on_train_begin()` method from every callbacks.
        """
        for callback in self.callbacks:
            callback.on_train_begin(self)

    def __before_epoch_training(self):
        """
        (private) Before epoch training phase.
        --------------------------------------
        This function will be called before `__training_step()`.
        By default, this function will call `on_epoch_begin()` method from every callbacks.
        """
        for callback in self.callbacks:
            callback.on_epoch_begin(self)

    def __after_training(self):
        """
        (private) After training phase.
        -------------------------------
        This function will be called after `__training()`.
        By default, this function will call `on_train_end()` method from every callbacks.
        """
        for callback in self.callbacks:
            callback.on_train_end(self)

    def __after_epoch_training(self):
        """
        (private) After epoch training phase.
        -------------------------------------
        This function will be called after `__training_step()`.
        By default, this function will call `on_epoch_end()` method from every callbacks.
        """
        for callback in self.callbacks:
            callback.on_epoch_end(self)

    def __training_step(self, epoch, scaler):
        """
        (private) Training step.
        ------------------------
        The steps are:
        1. Iterate over training data.
        2. Zero your gradients for every batch.
        3. Enable autocasting for forward pass.
        4. Perform backward pass and optimization using scaler.
        """
        logger_message = f'Training epoch {epoch}/{self.epochs}'
        avg_train_loss = torch.Tensor([0]).to(self.device)

        progress_bar = tqdm(enumerate(self.train_generator),
                            desc=logger_message, initial=0, dynamic_ncols=True)
        for _, data in progress_bar:
            # Destructuring data
            input_ids, attention_mask, token_type_ids, labels = data
            # Cast to device
            input_ids, attention_mask, token_type_ids, labels = cast_to_device(
                input_ids, attention_mask, token_type_ids, labels, device=self.device
            )

            # Zero your gradients for every batch
            self.optimizer.zero_grad(set_to_none=True)

            # Enable autocasting for forward pass
            with autocast():
                preds = self.model(input_ids, attention_mask, token_type_ids)

                # Middleware injection
                if self.middleware_preds is not None:
                    preds = self.middleware_preds(self, preds)

                loss = self.loss(preds, labels)
                avg_train_loss += loss

            # Perform backward pass and optimization using scaler
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            # Update progress bar
            progress_bar.set_description(
                f'{logger_message} | Loss: {loss.item():.4f}')

        return avg_train_loss / len(self.train_generator)

    def __validation_step(self, epoch):
        """
        (private) Validation step.
        --------------------------
        The steps are:
        1. Iterate over validation data.
        2. Disable gradient calculation for foward pass.
        """
        avg_valid_loss = torch.Tensor([0]).to(self.device)
        with torch.no_grad():
            for _, data in tqdm(enumerate(self.valid_generator), desc=f'Validating epoch {epoch}/{self.epochs}', initial=0, dynamic_ncols=True):
                # Destructuring data
                input_ids, attention_mask, token_type_ids, labels = data
                # Cast to device
                input_ids, attention_mask, token_type_ids, labels = cast_to_device(
                    input_ids, attention_mask, token_type_ids, labels, device=self.device
                )

                preds = self.model(
                    input_ids, attention_mask, token_type_ids)

                # Middleware injection
                if self.middleware_preds is not None:
                    preds = self.middleware_preds(self, preds)

                loss = self.loss(preds, labels)
                avg_valid_loss += loss

        return avg_valid_loss / len(self.valid_generator)

    def __save_checkpoint(self, epoch, checkpoint_name: str = None):
        """
        (private) Save checkpoint.
        --------------------------
        The checkpoint will be saved in `checkpoint_directory` with name `checkpoint_name`.
        If `checkpoint_name` is None, the checkpoint will be saved with name `next_checkpoint_name_id + epoch`.
        """
        if self.checkpoint_directory is not None:
            if checkpoint_name is None:
                checkpoint_name = f'{self.next_checkpoint_name_id + epoch}{CHECKPOINT_EXTENSION}'
            path = os.path.join(self.checkpoint_directory, checkpoint_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss_state_dict': self.loss.state_dict(),
            }, path)

    def __training(self):
        """
        (private) Training phase.
        -------------------------
        The steps are:
        1. Define scaler for loss scaling.
        2. Iterate over training data.
        3. Call `__before_epoch_training()`.
        4. Train model.
        5. Validate model.
        6. Save every checkpoint.
        7. Save best checkpoint.
        8. Call `__after_epoch_training()`.
        """
        # Define scaler for loss scaling
        scaler = GradScaler()

        # Iterate over training data
        for epoch in range(self.epochs):
            # Break if time limit reached
            if self.time_limit is not None and time.time() - self.training_start_time > self.time_limit:
                break

            # Before training
            self.__before_epoch_training()

            # Training
            self.model.train(True)
            avg_train_loss = self.__training_step(epoch, scaler)

            # Validate
            if self.valid_generator is not None:
                self.model.eval(True)
                avg_valid_loss = self.__validation_step(epoch)
            else:
                avg_valid_loss = 0

            # Save checkpoint
            self.__save_checkpoint(epoch)

            # Save best model
            if self.valid_generator is not None and avg_valid_loss < self.best_valid_loss:
                self.best_valid_loss = avg_valid_loss
                self.__save_checkpoint(epoch, f'0{CHECKPOINT_EXTENSION}')

            # Save history
            self.histories['train_loss'].append(avg_train_loss.item())
            self.histories['valid_loss'].append(avg_valid_loss.item())

            # After training
            self.__after_epoch_training()

    def fit(
        self,
        train_generator: Dataset,
        valid_generator: Dataset = None,
        batch_size: int = None,
        epochs: int = None,
        checkpoint_directory: str = None,
        callbacks: list[BaseCallback] = None,
        **kwargs
    ):
        """
        Fit model with given data generator.
        -------------------------------------
        Args:
            train_generator (torch.utils.data.Dataset): Training data generator.
            valid_generator (torch.utils.data.Dataset): Validation data generator.
            batch_size (int): Batch size.
            epochs (int): Epochs.
            checkpoint_directory (str): Directory to save model checkpoints.
            callbacks (list): List of callbacks to be used.

        Keyword Args:
            next_checkpoint_name_id (int): Next checkpoint name id. Default: 1.
            middleware_preds (function): Preprocessing prediction function. Default: None.

        The steps are:
        1. Set arguments.
        2. Call `__before_training()`.
        3. Call `__training()`.
        4. Call `__after_training()`.
        """
        # Set arguments
        self.train_generator = train_generator
        self.valid_generator = valid_generator
        self.batch_size = batch_size
        self.epochs = epochs
        self.checkpoint_directory = checkpoint_directory
        if callbacks is not None:
            self.callbacks = callbacks

        # Checker
        assert self.epochs is not None, "Epochs must be defined."

        # Define some default kwargs
        self.next_checkpoint_name_id = kwargs.get('next_checkpoint_name_id', 1)
        self.middleware_preds = kwargs.get('middleware_preds', None)

        # Before training
        self.__before_training()

        # Training
        self.training_start_time = time.time()
        self.__training()

        # After training
        self.__after_training()

    def load_weights(self, checkpoint_path: str):
        """
        Load model's weights from checkpoint.
        -------------------------------------
        Args:
            checkpoint_path (str): Path to checkpoint.
        """
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss.load_state_dict(checkpoint['loss_state_dict'])
