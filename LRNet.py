import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor, Compose, RandomCrop

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning import Trainer

import copy

from discrete import DiscreteLinear, DiscreteConv2d, DiscreteLayer


class LRNet(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.net = nn.Sequential(
            DiscreteConv2d(1, 32, 5, padding=2, binary=hparams['binary']),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            DiscreteConv2d(32, 64, 5, padding=2, binary=hparams['binary']),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.5),
            nn.Flatten(),

            DiscreteLinear(3136, 512, binary=hparams['binary']),
            nn.BatchNorm1d(512),

            nn.Linear(512, 10)
        )

        self.hparams = hparams
        self.loss = nn.CrossEntropyLoss()
        self.val_nets = None

    def to_discrete(self):
        layers = []
        for module in self.net.modules():
            if isinstance(module, DiscreteLayer):
                layers.append(module.to_discrete())
            else:
                layers.append(copy.deepcopy(module))

        return nn.Sequential(*layers).cuda()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        pred = self(batch[0])
        loss = self.loss(pred, batch[1])

        weight_decay = 0
        weight_decay_n = 0
        probability_decay = 0
        beta = 0

        for module in self.net.modules():
            if isinstance(module, DiscreteLayer):
                for param in module.parameters():
                    probability_decay += param.norm(2)

                probs = module.get_probs()
                for i in range(len(probs)):
                    beta += (1.0 / (probs[i] * (1 - probs[i]))).sum()

            else:
                for param in module.parameters():
                    weight_decay += param.norm(2)
                    weight_decay_n += 1

        weight_decay /= weight_decay_n

        loss += self.hparams['probability_decay'] * probability_decay + self.hparams['weight_decay'] * weight_decay + \
                beta * self.hparams['beta_parameter']

        correct = (torch.argmax(pred, 1) == batch[1]).sum()
        total = pred.size(0)

        return {'loss': loss, 'correct': correct, 'total': total}

    def training_epoch_end( self, outputs):
        correct = torch.stack([v['correct'] for v in outputs], 0).sum()
        total = float(sum([v['total'] for v in outputs]))

        logs = {
            'train_loss': torch.stack([v['loss'] for v in outputs], 0).mean(),
            'accuracy': correct / total
        }

        return {'log': logs}

    def train_dataloader(self):
        data = MNIST('data/', train=True, download=True, transform=ToTensor())
        return DataLoader(data, batch_size=256, shuffle=True)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])
        return [optim], [torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[100], gamma=0.1)]

    def val_dataloader(self):
        data = MNIST('data/', train=False, download=True, transform=ToTensor())
        return DataLoader(data, batch_size=256)

    def validation_step(self, batch, batch_idx):
        if self.val_nets is None:
            self.val_nets = [self.to_discrete().eval() for _ in range(self.hparams['num_val_nets'])]

        val_loss = []
        correct = []
        total = batch[0].size(0)

        for i in range(self.hparams['num_val_nets']):
            pred = self(batch[0])
            val_loss.append(self.loss(pred, batch[1]))
            correct.append((torch.argmax(pred, 1) == batch[1]).sum())

        val_loss = torch.stack(val_loss)
        correct = torch.stack(correct)

        return {'val_loss': val_loss, 'correct': correct, 'total': total}

    def validation_epoch_end(self, outputs):
        self.val_nets = None

        correct = torch.stack([v['correct'] for v in outputs], 1).sum(1)
        total = float(sum([v['total'] for v in outputs]))

        accs = correct / total

        loss = torch.stack([v['val_loss'] for v in outputs], 1).mean(1)

        logs = {
            'val_acc/mean': accs.mean(),
            'val_acc/std': accs.std(),
            'val_acc/min': accs.min(),
            'val_acc/max': accs.max(),

            'val_loss/mean': loss.mean(),
            'val_loss/std': loss.std(),
            'val_loss/min': loss.min(),
            'val_loss/max': loss.max()
        }

        return {'val_loss': loss.mean(), 'log': logs}


if __name__ == '__main__':
    '''
    Available hyperparameters:
    batch_size: number of training samples per iteration
    lr: learning rate
    num_val_nets: Number of sampled networks for the validation
    val_every_n: validate every n epochs
    epochs: number of epochs
    probability_decay: factor of the entropy increasing regularizer
    weight_decay: l2 regularizer of the last full precision layer
    beta_parameter: factor of the entropy decreasing regularizer
    binary: true for binary weights, false for ternary
    '''
    # example settings for ternary weights
    hparams = {
        'batch_size': 256,
        'lr': 0.01,
        'num_val_nets': 10,
        'val_every_n': 10,
        'epochs': 190,
        'probability_decay': 1e-11,
        'weight_decay': 1e-4,
        'beta_parameter': 0.0,
        'binary': False
    }

    # example settings for binary weights
    '''hparams = {
        'batch_size': 256,
        'lr': 0.01,
        'num_val_nets': 10,
        'val_every_n': 10,
        'epochs': 190,
        'probability_decay': 0.0,
        'weight_decay': 1e-4,
        'beta_parameter': 1e-6,
        'binary': True
    }'''

    model = LRNet(hparams)
    trainer = Trainer(gpus=1, max_epochs=hparams['epochs'], check_val_every_n_epoch=hparams['val_every_n'],
                      callbacks=[LearningRateLogger()])

    trainer.fit(model)
