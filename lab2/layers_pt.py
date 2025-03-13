import torch
import torch.nn as nn


class ConvolutionalModel(nn.Module):
    def __init__(self, input_size, n_classes):
        super().__init__()
        H, W, C = input_size
        H_out, W_out = H//4, W//4
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=16, kernel_size=(5,5), padding='same'),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5,5), padding='same'),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=32*H_out*W_out, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=n_classes)
        )
        
        # parametri su već inicijalizirani pozivima Conv2d i Linear
        # ali možemo ih drugačije inicijalizirati
        ### self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m is not self.fc_logits:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        self.fc_logits.reset_parameters()

    def forward(self, x):
        logits = self.layers(x)
        return logits
