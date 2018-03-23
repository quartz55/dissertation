from typing import Optional, Callable, Dict, List
from abc import ABCMeta, abstractmethod
from enum import Enum

import time
import math
import datetime
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader


class Events(Enum):
    EPOCH_STARTED = 'epoch_started'
    EPOCH_COMPLETED = 'epoch_completed'
    STARTED = 'started'
    COMPLETED = 'completed'
    ITERATION_STARTED = 'iteration_started'
    ITERATION_COMPLETED = 'iteration_completed'


class State(metaclass=ABCMeta):
    def __init__(self):
        self.iteration = 0
        self.output = None
        self.batch = None


class Engine(metaclass=ABCMeta):
    def __init__(self, process_fn: Callable):
        assert process_fn is not None, "A processing function must be provided in order for the Engine to run"
        self._logger: logging.Logger = logging.getLogger(
            __name__ + '.' + self.__class__.__name__
        )
        self._logger.addHandler(logging.NullHandler())
        self._event_handlers: Dict[Events, List[Callable]] = {}
        self._process_fn: Callable = process_fn
        self.state: Optional[State] = None

    def add_event_handler(self, event: Events, handler: Callable):
        if event not in self._event_handlers:
            self._event_handlers[event] = []

        self._event_handlers[event].append(handler)
        self._logger.debug("Added handler for event '{}'".format(event))

    def when(self, event: Events):
        def decorator(f):
            self.add_event_handler(event, f)
            return f

        return decorator

    def _trigger_event(self, event: Events):
        if event in self._event_handlers:
            self._logger.debug(
                "Triggering handlers for event '{}'".format(event)
            )
            for fn in self._event_handlers[event]:
                fn(self.state)

    def _iteration(self) -> int:
        start_time = time.time()
        for batch in self.state.dataloader:
            self.state.iteration += 1
            self.state.batch = batch
            self._trigger_event(Events.ITERATION_STARTED)
            self.state.output = self._process_fn(batch)
            self._trigger_event(Events.ITERATION_COMPLETED)
        time_taken = time.time() - start_time
        return time_taken

    @abstractmethod
    def run(self, dataloader: DataLoader):
        raise NotImplementedError


class TrainerState(State):
    def __init__(self, loader: DataLoader, max_epochs=10):
        super().__init__()
        self.dataloader = loader
        self.epoch = 0
        self.max_epochs = max_epochs


class Trainer(Engine):
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        use_cuda: bool = False
    ):
        self.model: nn.Module = model
        self.optimizer: optim.Optimizer = optimizer
        self.criterion: nn.Module = criterion
        self.use_cuda: bool = use_cuda

        def process_fn(batch):
            inputs, targets = batch
            self.model.train()
            self.optimizer.zero_grad()
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = self.model(inputs)
            loss: nn.Module = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            return {'outputs': outputs, 'loss': loss.data.cpu()[0]}

        super().__init__(process_fn)

    def run(self, loader: DataLoader, max_epochs=10):
        self.state = TrainerState(loader, max_epochs)
        self._logger.info(
            "Training model {} on {} samples (max epochs={})".format(
                self.model.__class__.__name__, len(loader.dataset), max_epochs
            )
        )
        start_time = time.time()
        self._trigger_event(Events.STARTED)
        while self.state.epoch < self.state.max_epochs:
            self.state.epoch += 1
            self._trigger_event(Events.EPOCH_STARTED)
            iter_time = self._iteration()
            self._logger.info(
                "Epoch {:{epoch_size}} completed ({})".format(
                    self.state.epoch,
                    datetime.timedelta(milliseconds=iter_time),
                    epoch_size=int(math.log10(self.state.max_epochs)) + 1
                )
            )
            self._trigger_event(Events.EPOCH_COMPLETED)
        self._trigger_event(Events.COMPLETED)
        time_taken = time.time() - start_time
        self._logger.info(
            "Training of model {} complete ({} epochs, {})".format(
                self.model.__class__.__name__,
                self.state.epoch,
                datetime.timedelta(milliseconds=time_taken)
            )
        )