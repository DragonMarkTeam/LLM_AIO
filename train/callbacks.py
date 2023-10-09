"""
Callbacks classes.
"""

from abc import abstractmethod


class BaseCallback:
    """
    Base class for all callbacks.
    """
    @abstractmethod
    def on_train_begin(self, trainer):
        """
        Called before training.
        """

    @abstractmethod
    def on_train_end(self, trainer):
        """
        Called after training.
        """

    @abstractmethod
    def on_epoch_begin(self, trainer):
        """
        Called before epoch training.
        """

    @abstractmethod
    def on_epoch_end(self, trainer):
        """
        Called after epoch training.
        """
