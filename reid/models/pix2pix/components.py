import torch
from torch import nn


class Up(nn.Sequential):
    def __init__(self, in_channels, out_channels, normalization=True, dropout=None):

        layers = [
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            )
        ]
        if normalization:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))

        super().__init__(*layers)

    def forward(self, x, skip):
        x = super().forward(x)
        x = torch.cat((x, skip), 1)
        return x


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels, normalization=True, dropout=None):

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        ]
        if normalization:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))

        super().__init__(*layers)


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.down1 = Down(in_channels, 64, normalization=False)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512, dropout=0.5)
        self.down5 = Down(512, 512, dropout=0.5)
        self.down6 = Down(512, 512, dropout=0.5)
        self.down7 = Down(512, 512, dropout=0.5)
        self.down8 = Down(512, 512, dropout=0.5, normalization=False)

        self.up1 = Up(512, 512, dropout=0.5)
        self.up2 = Up(1024, 512, dropout=0.5)
        self.up3 = Up(1024, 512, dropout=0.5)
        self.up4 = Up(1024, 512, dropout=0.5)
        self.up5 = Up(1024, 256)
        self.up6 = Up(512, 128)
        self.up7 = Up(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        down6 = self.down6(down5)
        down7 = self.down7(down6)
        down8 = self.down8(down7)

        up1 = self.up1(down8, down7)
        up2 = self.up2(up1, down6)
        up3 = self.up3(up2, down5)
        up4 = self.up4(up3, down4)
        up5 = self.up5(up4, down3)
        up6 = self.up6(up5, down2)
        up7 = self.up7(up6, down1)

        return self.final(up7)


class Discriminator(nn.Sequential):
    def __init__(self, in_channels = 3):
        super().__init__(
            Down(in_channels * 2, 64, normalization=False),
            Down(64, 128),
            Down(128, 256),
            Down(256, 512),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        return super().forward(x)
