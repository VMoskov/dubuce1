import torch
import torch.nn as nn


class ConvolutionalModel(nn.Module):
    def __init__(self, input_size, n_classes, batch_norm=False):
        super().__init__()
        H, W, C = input_size
        H_out, W_out = H//4, W//4

        self.conv1 = nn.Conv2d(in_channels=C, out_channels=16, kernel_size=5, padding='same')  # save for visualization
        self.layers = nn.Sequential(
            self.conv1,
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding='same'),
            nn.MaxPool2d(kernel_size=2),
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
    

class SimpleModel(nn.Module):
    def __init__(self, input_size, n_classes):
        super().__init__()
        H, W, C = input_size
        # H_out, W_out = H//4, W//4

        self.conv1 = nn.Conv2d(in_channels=C, out_channels=32, kernel_size=5, padding='same')  # save for visualization
        self.conv_layers = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # calculate the size of the output of the conv layers
        conv_out_size = self._output_size(H, W, C)  

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=conv_out_size, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=n_classes)
        )
        self.layers = nn.Sequential(
            self.conv_layers,
            self.fc
        )

    def forward(self, x):
        return self.layers(x)
        
    def _output_size(self, H, W, C):
        dummy_input = torch.zeros(1, C, H, W)
        dummy_output = self.conv_layers(dummy_input)
        return dummy_output.size(1) * dummy_output.size(2) * dummy_output.size(3)
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        padding = kernel_size // 2  # manually computing the 'same' padding since nn.Conv2d doesn't support it with strided conv

        # BN -> ReLU -> Conv as proposed in https://arxiv.org/abs/1512.03385
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding='same')
        )
        if in_channels != out_channels or downsample:  # if dimensions change, need to adjust residual connection
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:  # if dimensions don't change, residual connection is identity
            self.residual = nn.Identity()
        
    def forward(self, x):
        skip_connection = self.residual(x)
        x = self.layers(x)
        return x + skip_connection
    

class ResidualModel(nn.Module):
    def __init__(self, input_size, n_classes):
        super().__init__()
        H, W, C = input_size
        
        self.conv1 = nn.Conv2d(in_channels=C, out_channels=16, kernel_size=5, padding='same')  # save for visualization
        self.init_conv = nn.Sequential(
            self.conv1,
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        H_out, W_out = H//2, W//2

        self.residual_blocks = nn.Sequential(
            ResidualBlock(in_channels=16, out_channels=32, kernel_size=5, downsample=True),
            ResidualBlock(in_channels=32, out_channels=64, kernel_size=5, downsample=True),
            ResidualBlock(in_channels=64, out_channels=128, kernel_size=5, downsample=True),
        )
        # dinamically compute the size of the output of the residual blocks
        residual_out_size = self._residual_output_size(H_out, W_out, C=16)  

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=residual_out_size, out_features=n_classes),
        )
        

    def forward(self, x):
        x = self.init_conv(x)
        x = self.residual_blocks(x)
        logits = self.fc(x)
        return logits

    def _residual_output_size(self, H, W, C):
        dummy_input = torch.zeros(1, C, H, W)
        dummy_output = self.residual_blocks(dummy_input)
        return dummy_output.size(1) * dummy_output.size(2) * dummy_output.size(3)