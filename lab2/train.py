import time
from pathlib import Path

import numpy as np
from torchvision.datasets import MNIST

import nn
import layers

DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
SAVE_DIR = Path(__file__).parent / 'out' / 'non_regularized'

# create directories if they don't exist
for dir_ in [DATA_DIR, SAVE_DIR]:
    dir_.mkdir(parents=True, exist_ok=True)

config = {}
config['max_epochs'] = 8
config['batch_size'] = 50
config['save_dir'] = SAVE_DIR
config['lr_policy'] = {1:{'lr':1e-1}, 3:{'lr':1e-2}, 5:{'lr':1e-3}, 7:{'lr':1e-4}}

def dense_to_one_hot(y, class_count):
    return np.eye(class_count)[y]

#np.random.seed(100) 
np.random.seed(int(time.time() * 1e6) % 2**31)

ds_train, ds_test = MNIST(DATA_DIR, train=True, download=True), MNIST(DATA_DIR, train=False)
train_x = ds_train.data.reshape([-1, 1, 28, 28]).numpy().astype(np.float32) / 255
train_y = ds_train.targets.numpy()
train_x, valid_x = train_x[:55000], train_x[55000:]
train_y, valid_y = train_y[:55000], train_y[55000:]
test_x = ds_test.data.reshape([-1, 1, 28, 28]).numpy().astype(np.float32) / 255
test_y = ds_test.targets.numpy()
train_mean = train_x.mean()
train_x, valid_x, test_x = (x - train_mean for x in (train_x, valid_x, test_x))
train_y, valid_y, test_y = (dense_to_one_hot(y, 10) for y in (train_y, valid_y, test_y))


net = []
inputs = np.random.randn(config['batch_size'], 1, 28, 28)
net += [layers.Convolution(input_layer=inputs, num_filters=16, kernel_size=5, name="conv1")]
net += [layers.MaxPooling(input_layer=net[-1], name="pool1")]
net += [layers.ReLU(input_layer=net[-1], name="relu1")]
net += [layers.Convolution(input_layer=net[-1], num_filters=32, kernel_size=5, name="conv2")]
net += [layers.MaxPooling(input_layer=net[-1], name="pool2")]
net += [layers.ReLU(input_layer=net[-1], name="relu2")]
# out = 7x7
net += [layers.Flatten(input_layer=net[-1], name="flatten3")]
net += [layers.FC(input_layer=net[-1], num_outputs=512, name="fc3")]
net += [layers.ReLU(input_layer=net[-1], name="relu3")]
net += [layers.FC(input_layer=net[-1], num_outputs=10, name="logits")]

loss = layers.SoftmaxCrossEntropyWithLogits()

nn.train(train_x, train_y, valid_x, valid_y, net, loss, config)
nn.evaluate("Test", test_x, test_y, net, loss, config)