import time
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

from layers_pt import ConvolutionalModel, ResidualModel, SimpleModel, FMPModel
from convpt_utils import train, evaluate, show_highest_loss_images


DATA_DIR = Path(__file__).parent / 'datasets' / 'CIFAR10'
SAVE_DIR = Path(__file__).parent / 'out' / 'pt_conv' / 'CIFAR10'

# create directories if they don't exist
for dir_ in [DATA_DIR, SAVE_DIR]:
    dir_.mkdir(parents=True, exist_ok=True)

config = {}
config['max_epochs'] = 50
# config['batch_size'] = 50
config['batch_size'] = 64
config['save_dir'] = SAVE_DIR
config['weight_decay'] = 1e-4
# config['weight_decay'] = 1e-3
# config['weight_decay'] = 1e-2
# config['weight_decay'] = 1e-1

np.random.seed(int(time.time() * 1e6) % 2**31)

mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)  # https://github.com/kuangliu/pytorch-cifar/issues/19

# scale images to higher resolution
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),   
    transforms.RandomCrop(size=64, padding=8),
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomRotation(degrees=15),  
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)), 
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# No augmentations for test/validation (only resizing & normalization)
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

trainset = CIFAR10(DATA_DIR, train=True, download=True, transform=train_transform)
testset = CIFAR10(DATA_DIR, train=False, transform=test_transform)

label_names = trainset.classes

train_size = len(trainset)
valid_size = train_size // 10
train_size -= valid_size
trainset, valset = torch.utils.data.random_split(trainset, [train_size, valid_size])

trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
valloader = DataLoader(valset, batch_size=config['batch_size'], shuffle=False)
testloader = DataLoader(testset, batch_size=config['batch_size'], shuffle=False)

weight_decay = config['weight_decay']


if __name__ == '__main__':
    weight_decay_str = format(weight_decay, '.0e')
    SAVE_DIR = SAVE_DIR / weight_decay_str
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    model = ResidualModel(input_size=(64, 64, 3), n_classes=10)
    # model = SimpleModel(input_size=(64, 64, 3), n_classes=10)
    # model = FMPModel(input_size=(64, 64, 3), n_classes=10)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['weight_decay'])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    model, losses = train(model, criterion, optimizer, scheduler, trainloader, valloader, config, SAVE_DIR)
    accuracy, _ = evaluate(model, testloader, criterion)
    print(f'Test accuracy: {accuracy}')

    # plot losses
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training loss, weight decay: {weight_decay}')
    plt.savefig(SAVE_DIR.parent / f'{weight_decay_str}_loss.png')
    plt.show()

    # show highest loss images
    show_highest_loss_images(model, testloader, label_names, mean, std)