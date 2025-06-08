from torch.utils.data import Dataset
from collections import defaultdict
from random import choice
import torchvision


class MNISTMetricDataset(Dataset):
    def __init__(self, root='/tmp/mnist/', split='train', remove_class=None):
        super().__init__()
        assert split in ['train', 'test', 'traineval']
        self.root = root
        self.split = split
        mnist_ds = torchvision.datasets.MNIST(self.root, train='train' in split, download=True)
        self.images, self.targets = mnist_ds.data.float().unsqueeze(1) / 255., mnist_ds.targets
        self.classes = list(range(10))

        if remove_class is not None:
            assert remove_class in self.classes, f"Class {remove_class} not in MNIST classes {self.classes}"
            mask = self.targets != remove_class
            self.images = self.images[mask]
            self.targets = self.targets[mask]

        self.target2indices = defaultdict(list)
        for i in range(len(self.images)):
            self.target2indices[self.targets[i].item()] += [i]

    def _sample_negative(self, index):
        while True:
            negative_index = choice(range(len(self.images)))
            if negative_index != index and self.targets[negative_index] != self.targets[index]:
                return negative_index


    def _sample_positive(self, index):
        target_id = self.targets[index].item()
        positive_indices = self.target2indices[target_id]
        while True:
            positive_index = choice(positive_indices)
            if positive_index != index:
                return positive_index


    def __getitem__(self, index):
        anchor = self.images[index]
        target_id = self.targets[index].item()
        if self.split in ['traineval', 'val', 'test']:
            return anchor, target_id
        else:
            positive = self._sample_positive(index)
            negative = self._sample_negative(index)
            positive = self.images[positive]
            negative = self.images[negative]
            return anchor, positive, negative, target_id

    def __len__(self):
        return len(self.images)
