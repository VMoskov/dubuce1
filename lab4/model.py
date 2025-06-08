import torch
import einops
import torch.nn as nn
import torch.nn.functional as F


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(num_maps_in))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_maps_in, num_maps_out, kernel_size=k, padding=k // 2, bias=bias))


class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32):
        super().__init__()
        self.emb_size = emb_size
        # YOUR CODE HERE
        self.layers = nn.Sequential(
            _BNReluConv(num_maps_in=input_channels, num_maps_out=self.emb_size, k=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            _BNReluConv(num_maps_in=self.emb_size, num_maps_out=self.emb_size, k=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            _BNReluConv(num_maps_in=self.emb_size, num_maps_out=self.emb_size, k=3),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )

    def get_features(self, img):
        x = self.layers(img)
        x = einops.rearrange(x, 'b c h w -> b (c h w)')  # flatten to (batch_size, emb_size)
        return x

    def loss(self, anchor, positive, negative):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)

        a_x = F.normalize(a_x, dim=1)
        p_x = F.normalize(p_x, dim=1)
        n_x = F.normalize(n_x, dim=1)

        # loss = F.triplet_margin_loss(anchor=a_x, positive=p_x, negative=n_x)

        # implementing triplet loss manually
        margin = 1.0
        pos_dist = F.pairwise_distance(a_x, p_x, keepdim=True)
        neg_dist = F.pairwise_distance(a_x, n_x, keepdim=True)
        loss = F.relu(pos_dist - neg_dist + margin).mean()
        return loss
    

class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()
        # no trainable parameters
        self.dummy_param = nn.Parameter(torch.empty(0))

    def get_features(self, img):
        # just return the flattened image input as features
        feats = einops.rearrange(img, 'b c h w -> b (c h w)')  # flatten to (batch_size, num_features)
        return feats
    
    def loss(self, anchor, positive, negative):
        # Identity model does not use triplet loss, just return zero loss
        return torch.tensor(0.0, device=anchor.device) + 0*self.dummy_param.sum()