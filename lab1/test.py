import numpy as np
import torch
from torch.utils.data import DataLoader

dataset = [(torch.randn(4, 4), torch.randint(5, size=())) for _ in range(25)]

print(dataset)
