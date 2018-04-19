from typing import Dict, Any

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from .engine import Engine, Events
from ..metrics.metric import EngineMetric


class Trainer(Engine):
    def __init__(self,
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 criterion: nn.Module,
                 use_cuda: bool = False) -> None:
        super().__init__()
        self.model: nn.Module = model
        self.optimizer: optim.Optimizer = optimizer
        self.criterion: nn.Module = criterion
        self.use_cuda: bool = use_cuda
        self._process_fn = self.update_model

    def update_model(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        inputs, targets = batch
        if self.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = self.model(inputs)
        loss: nn.Module = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return {'outputs': outputs, 'loss': float(loss.data.cpu()[0])}


class Evaluator(Engine):
    def __init__(self, model: nn.Module, use_cuda: bool = False) -> None:
        super().__init__()
        self.model: nn.Module = model
        self.use_cuda: bool = use_cuda
        self.metrics: Dict[str, Any] = {}
        self._process_fn = self.inference
        self.__metrics: Dict[str, EngineMetric] = {}
        self.add_event_handler(Events.EPOCH_COMPLETED, self.__compute_metrics)

    def inference(self, batch):
        self.model.eval()
        inputs, targets = batch
        if self.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        prediction = self.model(inputs)
        return prediction.data.cpu(), targets.data.cpu()

    def add_metric(self, metric: EngineMetric, name: str = None):
        if name is None:
            name = metric.__class__.__name__
        metric.attach(self)
        self.__metrics[name] = metric

    def __compute_metrics(self, _: Engine):
        for name, metric in self.__metrics.items():
            self.metrics[name] = metric.compute()