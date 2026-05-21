import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels, n_classes, inc_channels, interpolation='nearest'):
        super(UNet, self).__init__()
        self.inc = DoubleConv(in_channels, inc_channels)

        self.down1 = Down(inc_channels, inc_channels * 2)
        self.down2 = Down(inc_channels * 2, inc_channels * 4)
        self.down3 = Down(inc_channels * 4, inc_channels * 8)
        self.down4 = Down(inc_channels * 8, inc_channels * 8)

        self.up4 = Up(inc_channels * 16, inc_channels * 4, interpolation)
        self.up3 = Up(inc_channels * 8, inc_channels * 2, interpolation)
        self.up2 = Up(inc_channels * 4, inc_channels, interpolation)
        self.up1 = Up(inc_channels * 2, inc_channels, interpolation)

        self.outc = OutConv(inc_channels, n_classes)

    def forward(self, x):
        d0 = self.inc(x)  # 32

        d1 = self.down1(d0)  # 64
        d2 = self.down2(d1)  # 128
        d3 = self.down3(d2)  # 256
        d4 = self.down4(d3)  # 256

        u4 = self.up4(d4, d3)  # 128
        u3 = self.up3(u4, d2)  # 64
        u2 = self.up2(u3, d1)  # 32
        u1 = self.up1(u2, d0)  # 32

        u0 = self.outc(u1)  # 3

        return u0


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Up-scaling then double conv"""

    def __init__(self, in_channels, out_channels, interpolation='nearest'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode=interpolation)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        out = self.conv(x)
        return out


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
