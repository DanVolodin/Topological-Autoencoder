import torch
import torch.nn as nn
import torch.nn.functional as F
    
class BasicAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.maxpool = nn.MaxPool2d(2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv1(x)
        x_encoded = self.conv2(self.maxpool(y))
        x_decoded = self.upconv1(x_encoded) + y
        return self.conv3(x_decoded)


class ModifiedAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.upconv = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2)

        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.upconv(self.encoder(x))

        diffX = x.size()[2] - y.size()[2]
        diffY = x.size()[3] - y.size()[3]
        y = F.pad(y, (diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2))

        return self.decoder(y)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_up = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.unet = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=False)
        self.conv_down = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv_down(self.unet(self.conv_up(x))))
    