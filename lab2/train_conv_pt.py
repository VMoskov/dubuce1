import time
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import cv2
import matplotlib.pyplot as plt

from layers_pt import ConvolutionalModel


DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
SAVE_DIR = Path(__file__).parent / 'out' / 'pt_conv'

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

def dense_to_one_hot(y, class_count):
    return np.eye(class_count)[y]

np.random.seed(int(time.time() * 1e6) % 2**31)

transorm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

trainset = MNIST(DATA_DIR, train=True, download=True, transform=transorm)
testset = MNIST(DATA_DIR, train=False, transform=transorm)

train_size = len(trainset)
valid_size = train_size // 12
train_size -= valid_size
trainset, valset = torch.utils.data.random_split(trainset, [train_size, valid_size])

trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
valloader = DataLoader(valset, batch_size=config['batch_size'], shuffle=False)
testloader = DataLoader(testset, batch_size=config['batch_size'], shuffle=False)

weight_decay = config['weight_decay']


def draw_conv_filters(epoch, step, conv_layer, save_dir):
    weights = conv_layer.weight.data.cpu().numpy()
    num_filters = weights.shape[0]
    num_channels = weights.shape[1]
    k = weights.shape[2]

    assert weights.shape[2] == weights.shape[3]  # Ensure square filters
    
    num_rows, num_cols = 2, 8  
    assert num_filters >= num_rows * num_cols, "Not enough filters to fill the grid"

    border_size = 1  
    filter_images = []
    
    for i in range(num_rows * num_cols):
        img = weights[i, 0, :, :]  
        img -= img.min()
        img /= img.max()
        img = (img * 255).astype(np.uint8)
        img = np.repeat(np.expand_dims(img, 2), 3, axis=2)  
        filter_images.append(img)
    
    filter_h, filter_w, _ = filter_images[0].shape

    # Create a blank black canvas with separators
    canvas_h = num_rows * filter_h + (num_rows - 1) * border_size
    canvas_w = num_cols * filter_w + (num_cols - 1) * border_size
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for row in range(num_rows):
        for col in range(num_cols):
            x = col * (filter_w + border_size)
            y = row * (filter_h + border_size)
            canvas[y:y + filter_h, x:x + filter_w] = filter_images[row * num_cols + col]

    cv2.imwrite(f'{save_dir}/conv1_epoch_{epoch:02d}_step_{step:06d}_input_000.png', canvas)


def train(model, criterion, optimizer, trainloader, valloader, config):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    max_epochs = config['max_epochs']
    batch_size = config['batch_size']
    lr_policy = config['lr_policy']

    losses = []
    for epoch in range(1, max_epochs + 1):
        epoch_loss = 0
        model.train()

        if epoch in lr_policy:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_policy[epoch]['lr']

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                draw_conv_filters(epoch, i*batch_size, model.layers[0], SAVE_DIR)

        accuracy = evaluate(model, valloader)
        print(f'Epoch {epoch}, validation accuracy: {accuracy}, loss: {epoch_loss}')
        losses.append(epoch_loss)

    return model, losses


def evaluate(model, dataloader):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


if __name__ == '__main__':
    SAVE_DIR = SAVE_DIR / format(weight_decay, '.0e')
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    model = ConvolutionalModel(input_size=(28, 28, 1), n_classes=10)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['weight_decay'])

    model, losses = train(model, criterion, optimizer, trainloader, valloader, config)
    accuracy = evaluate(model, testloader)
    print(f'Test accuracy: {accuracy}')

    # plot losses
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training loss, weight decay: {weight_decay}')
    plt.savefig(SAVE_DIR.parent / f'{format(weight_decay, '.0e')}_loss.png')
    plt.show()