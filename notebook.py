# %%
# %load_ext autoreload
# %autoreload 2

# %%
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tqdm import tqdm
import numpy as np
import visdom as visdom

from cimc.supervisors import Trainer, Events, TrainerState
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


# %%
def get_mnist_loaders(train_batch_size=64, validation_batch_size=64):
    transf = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))]
    )

    train_dset = datasets.MNIST(
        root='datasets/mnist', train=True, download=True, transform=transf
    )
    test_dset = datasets.MNIST(
        root='datasets/mnist', train=False, transform=transf
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dset,
        batch_size=validation_batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    return train_loader, test_loader


def visdom_plot(vis: visdom.Visdom, xlabel, ylabel, title):
    return vis.line(
        X=np.array([1]),
        Y=np.array([np.nan]),
        opts={
            'xlabel': xlabel,
            'ylabel': ylabel,
            'title': title
        }
    )


# %%
use_cuda = True
CUDA = use_cuda and torch.cuda.is_available()
train_loader, test_loader = get_mnist_loaders(64, 1000)

# %%
model = Mnist4Layers()
if CUDA:
    model.cuda()
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# %%
vis = visdom.Visdom()
if not vis.check_connection():
    raise RuntimeError(
        "Visdom server isn't running! Please run 'python -m visdom.server'"
    )

train_loss_plot = visdom_plot(vis, '#Iterations', 'Loss', 'Training Loss')
val_acc_plot = visdom_plot(vis, '#Epochs', 'Accuracy', 'Validation Accuracy')
val_loss_plot = visdom_plot(vis, '#Iterations', 'Loss', 'Validation Loss')

# %%
trainer = Trainer(model, optimizer, criterion, use_cuda=CUDA)

iter_bar = tqdm(total=len(train_loader.dataset), unit='sample')
losses = []


@trainer.when(Events.ITERATION_COMPLETED)
def log_iteration_progress(state: TrainerState):
    inputs, _ = state.batch
    iter_bar.set_postfix(batch=state.iteration, refresh=False)
    iter_bar.update(len(inputs))
    losses.append(state.output['loss'])
    if (state.iteration - 1) % 10 == 0:
        mean_loss = np.mean(losses)
        losses.clear()
        iter_bar.set_description(
            "Epoch {} | Loss: {:.6f}".format(state.epoch, mean_loss)
        )
        vis.line(
            X=np.array([state.iteration]),
            Y=np.array([mean_loss]),
            win=train_loss_plot,
            update='append'
        )


@trainer.when(Events.EPOCH_COMPLETED)
def validate_results(state: TrainerState):
    global iter_bar
    iter_bar.close()
    print("Epoch {} done! Evaluating model here...".format(state.epoch))
    iter_bar = tqdm(total=len(train_loader.dataset), unit='sample')


# %%
logger.info('Running example trainer')
trainer.run(train_loader, 2)
