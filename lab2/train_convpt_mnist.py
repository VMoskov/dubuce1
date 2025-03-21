import time
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

from layers_pt import ConvolutionalModel
from convpt_utils import train, evaluate


DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
SAVE_DIR = Path(__file__).parent / 'out' / 'pt_conv' / 'MNIST'

# create directories if they don't exist
for dir_ in [DATA_DIR, SAVE_DIR]:
    dir_.mkdir(parents=True, exist_ok=True)

config = {}
config['max_epochs'] = 8
config['batch_size'] = 50
config['save_dir'] = SAVE_DIR
# config['weight_decay'] = 1e-3
# config['weight_decay'] = 1e-2
config['weight_decay'] = 1e-1
config['lr_policy'] = {1:{'lr':1e-1}, 3:{'lr':1e-2}, 5:{'lr':1e-3}, 7:{'lr':1e-4}}

np.random.seed(int(time.time() * 1e6) % 2**31)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

trainset = MNIST(DATA_DIR, train=True, download=True, transform=transform)
testset = MNIST(DATA_DIR, train=False, transform=transform)

train_size = len(trainset)
valid_size = train_size // 12
train_size -= valid_size
trainset, valset = torch.utils.data.random_split(trainset, [train_size, valid_size])

trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
valloader = DataLoader(valset, batch_size=config['batch_size'], shuffle=False)
testloader = DataLoader(testset, batch_size=config['batch_size'], shuffle=False)

weight_decay = config['weight_decay']


if __name__ == '__main__':
    SAVE_DIR = SAVE_DIR / format(weight_decay, '.0e')
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    model = ConvolutionalModel(input_size=(28, 28, 1), n_classes=10)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['weight_decay'])

    model, losses = train(model, criterion, optimizer, trainloader, valloader, config, SAVE_DIR)
    accuracy = evaluate(model, testloader)
    print(f'Test accuracy: {accuracy}')

    # plot losses
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training loss, weight decay: {weight_decay}')
    plt.savefig(SAVE_DIR.parent / f'{format(weight_decay, '.0e')}_loss.png')
    plt.show()