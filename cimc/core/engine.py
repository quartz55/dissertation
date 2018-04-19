from typing import Optional, Callable, Dict, List, Any, Tuple
from enum import Enum

import time
import math
import datetime
import logging

from torch.utils.data import DataLoader


class Events(Enum):
    EPOCH_STARTED = 'epoch_started'
    EPOCH_COMPLETED = 'epoch_completed'
    STARTED = 'started'
    COMPLETED = 'completed'
    ITERATION_STARTED = 'iteration_started'
    ITERATION_COMPLETED = 'iteration_completed'


class State:
    def __init__(self, data: DataLoader, max_epochs: int = 1) -> None:
        self.max_epochs: int = max_epochs
        self.iteration: int = 0
        self.epoch: int = 0
        self.output: Optional[Any] = None
        self.data: DataLoader = data
        self.batch: Optional[Any] = None


class Engine:
    def __init__(self, process_fn: Callable = lambda x: x) -> None:
        self._event_handlers: Dict[Events, List[Tuple[Callable, Any]]] = {}
        self.state: Optional[State] = None
        self._process_fn: Callable = process_fn
        self._logger: logging.Logger = logging.getLogger(
            __name__ + '.' + self.__class__.__name__
        )
        self._logger.addHandler(logging.NullHandler())

    def add_event_handler(self, event: Events, handler: Callable, *args):
        if event not in self._event_handlers:
            self._event_handlers[event] = []

        self._event_handlers[event].append((handler, args))
        self._logger.debug("Added handler for event '{}'".format(event))

    def when(self, event: Events, *args):
        def decorator(f):
            self.add_event_handler(event, f, *args)
            return f

        return decorator

    def _trigger_event(self, event: Events):
        if event in self._event_handlers:
            self._logger.debug(
                "Triggering handlers for event '{}'".format(event)
            )
            for (fn, args) in self._event_handlers[event]:
                fn(self, *args)

    def _iterate(self) -> float:
        start_time = time.time()
        for batch in self.state.data:
            self.state.iteration += 1
            self.state.batch = batch
            self._trigger_event(Events.ITERATION_STARTED)
            self.state.output = self._process_fn(batch)
            self._trigger_event(Events.ITERATION_COMPLETED)
        time_taken = time.time() - start_time
        return time_taken

    def run(self, dataloader: DataLoader, max_epochs: int = 1):
        self.state = State(dataloader, max_epochs)
        self._logger.info(
            "Running engine on dataset with {} samples (max epochs={})".format(
                len(dataloader.dataset), max_epochs
            )
        )
        start_time = time.time()
        self._trigger_event(Events.STARTED)
        while self.state.epoch < self.state.max_epochs:
            self.state.epoch += 1
            self._trigger_event(Events.EPOCH_STARTED)
            iter_time = self._iterate()
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
            "Engine stopped after {} epochs ({})".format(
                self.state.epoch, datetime.timedelta(milliseconds=time_taken)
            )
        )
