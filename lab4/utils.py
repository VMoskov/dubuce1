import numpy as np
import torch

from dataset import MNISTMetricDataset
from model import SimpleMetricEmbedding
from matplotlib import pyplot as plt


def get_colormap():
    # Cityscapes colormap for first 10 classes
    colormap = np.zeros((10, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    return colormap


def visualize_embeddings(test_repr, Y, colormap=get_colormap()):
    test_img_rep2d = torch.pca_lowrank(test_repr, q=2)[0].numpy()
    plt.scatter(test_img_rep2d[:, 0], test_img_rep2d[:, 1], c=colormap[Y[:]] / 255., s=5)
    plt.title('PCA of MNIST images')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=str(i),
               markerfacecolor=colormap[i] / 255., markersize=5) for i in range(10)],
               title='Classes', loc='upper right')
    plt.show()
    
    