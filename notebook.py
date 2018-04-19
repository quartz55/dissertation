# %load_ext autoreload
# %autoreload 2

import logging

import torch.cuda
import torch.utils.data
import torch.nn as nn
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tqdm import tqdm
import numpy as np
import visdom as visdom

from cimc.core import Events
from cimc.core.supervisors import Trainer, Evaluator
from cimc.metrics import Loss, CategoricalAccuracy
from cimc.models import Mnist4Layers


class TqdmHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)  # , file=sys.stderr)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            self.handleError(record)


logging.basicConfig(level=logging.INFO, handlers=[TqdmHandler()])
logger = logging.getLogger(__name__)


def get_mnist_loaders(train_batch_size=64, validation_batch_size=64):
    transf = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])

    train_dset = datasets.MNIST(
        root='datasets/mnist', train=True, download=True, transform=transf)
    test_dset = datasets.MNIST(
        root='datasets/mnist', train=False, transform=transf)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dset,
        batch_size=validation_batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True)

    return train_loader, test_loader


def visdom_plot(vis: visdom.Visdom, xlabel, ylabel, title):
    return vis.line(
        X=np.array([1]),
        Y=np.array([np.nan]),
        opts={
            'xlabel': xlabel,
            'ylabel': ylabel,
            'title': title
        })


class Experiment:
    def __init__(self, loaders, model, optimizer, criterion, use_cuda=True):
        self.cuda = use_cuda and torch.cuda.is_available()
        self.train_loader = loaders[0]
        self.test_loader = loaders[1]
        if self.cuda:
            model.cuda()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.vis = visdom.Visdom()
        if not self.vis.check_connection():
            raise RuntimeError(
                "Visdom server isn't running! Please run 'python -m visdom.server'"
            )
        self.state = {}

    def run(self, max_epochs=10):
        self.state = {
            'plots': {
                'train_loss':
                visdom_plot(self.vis, '#Iterations', 'Loss', 'Training Loss'),
                'val_loss':
                visdom_plot(self.vis, '#Epochs', 'Loss', 'Validation Loss'),
                'val_acc':
                visdom_plot(self.vis, '#Epochs', 'Accuracy',
                            'Validation Accuracy')
            },
            'trainer':
            Trainer(
                self.model, self.optimizer, self.criterion,
                use_cuda=self.cuda),
            'evaluator':
            Evaluator(self.model, use_cuda=self.cuda),
            'losses': [],
            'progress_bar':
            None
        }
        self.state['evaluator'].add_metric(Loss(self.criterion), 'cel')
        self.state['evaluator'].add_metric(CategoricalAccuracy(), 'acc')
        self.state['trainer'].add_event_handler(Events.EPOCH_STARTED,
                                                self.start_epoch)
        self.state['trainer'].add_event_handler(Events.ITERATION_COMPLETED,
                                                self.iteration)
        self.state['trainer'].add_event_handler(Events.EPOCH_COMPLETED,
                                                self.validation)
        self.state['trainer'].run(self.train_loader, max_epochs)

    def start_epoch(self, _):
        self.state['losses'].clear()
        self.state['progress_bar'] = tqdm(
            total=len(self.train_loader.dataset), unit='sample')

    def iteration(self, engine):
        inputs, _ = engine.state.batch
        bar = self.state['progress_bar']
        bar.set_postfix(batch=engine.state.iteration, refresh=False)
        bar.update(len(inputs))
        self.state['losses'].append(engine.state.output['loss'])
        if (engine.state.iteration - 1) % 50 == 0:
            mean_loss = np.mean(self.state['losses'])
            self.state['losses'].clear()
            bar.set_description("Epoch {} | Loss: {:.6f}".format(
                engine.state.epoch, mean_loss))
            self.vis.line(
                X=np.array([engine.state.iteration]),
                Y=np.array([mean_loss]),
                win=self.state['plots']['train_loss'],
                update='append')

    def validation(self, engine):
        self.state['progress_bar'].close()
        logger.info("Epoch {} done! Evaluating model here...".format(
            engine.state.epoch))
        self.state['evaluator'].run()
        logger.info("Validation loss: {:.6f}".format(
            self.state['evaluator'].metrics['cel']))
        self.vis.line(
            X=np.array([engine.state.epoch]),
            Y=np.array([self.state['evaluator'].metrics['cel']]),
            win=self.state['plots']['val_loss'],
            update='append')
        self.vis.line(
            X=np.array([engine.state.epoch]),
            Y=np.array([self.state['evaluator'].metrics['acc']]),
            win=self.state['plots']['val_acc'],
            update='append')


def main():
    train_loader, test_loader = get_mnist_loaders(64, 1000)
    model = Mnist4Layers()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    experiment = Experiment(
        (train_loader, test_loader),
        model,
        optimizer,
        criterion,
        use_cuda=True)
    logger.info('Training example MNIST network')
    experiment.run(10)


if __name__ == '__main__':
    main()
