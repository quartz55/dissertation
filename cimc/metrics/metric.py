from abc import ABCMeta, abstractmethod
from typing import Optional, Callable, List, Tuple

import torch
from torch.autograd import Variable
import numpy as np

from ..core import Events, Engine


class Metric(metaclass=ABCMeta):
    def __init__(self, transform: Optional[Callable]) -> None:
        if transform is None:
            transform = lambda x: x
        self._transform_fn = transform

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, batch_output):
        pass

    @abstractmethod
    def compute(self):
        pass


class EngineMetric(Metric, metaclass=ABCMeta):
    def __init__(self, engine: Engine = None,
                 transform: Callable = None) -> None:
        super().__init__(transform)
        if engine is not None:
            self.attach(engine)

    def _iteration(self, engine: Engine):
        output = self._transform_fn(engine.state.output)
        self.update(output)

    def attach(self, engine: Engine):
        engine.add_event_handler(Events.EPOCH_STARTED, lambda _: self.reset())
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._iteration)


class Loss(EngineMetric):
    def __init__(self,
                 loss_fn: Callable,
                 engine: Engine = None,
                 transform: Callable = None) -> None:
        super().__init__(engine, transform)
        self._loss_fn: Callable = loss_fn
        self._losses: List[float] = []

    def reset(self):
        self._losses.clear()

    def update(self, batch_output):
        y_pred, y = batch_output
        y_pred, y = Variable(y_pred, volatile=True), Variable(y, volatile=True)
        loss = self._loss_fn(y_pred, y)
        assert loss.shape == (1, )
        self._losses.append(loss.data[0])

    def compute(self):
        return np.mean(self._losses)


class CategoricalAccuracy(EngineMetric):
    BatchType = Tuple[torch.Tensor, torch.Tensor]

    def __init__(self, engine: Engine = None,
                 transform: Callable = None) -> None:
        super().__init__(engine, transform)
        self._correct: int = 0
        self._examples: int = 0

    def reset(self):
        self._correct = 0
        self._examples = 0

    def update(self, batch_output: BatchType):
        y_pred, y = batch_output
        indices = torch.max(y_pred, 1)[1]
        correct = torch.eq(indices, y).view(-1)
        self._correct += torch.sum(correct)
        self._examples += correct.shape[0]

    def compute(self):
        return self._correct / self._examples if self._examples > 0 else 0