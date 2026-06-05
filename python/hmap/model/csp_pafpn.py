import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, activation=True):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True) if activation else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class Bottleneck(nn.Module):
    def __init__(self, channels, expansion=0.5, shortcut=True):
        super().__init__()
        hidden = max(1, int(channels * expansion))
        self.conv1 = ConvBNAct(channels, hidden, kernel_size=1)
        self.conv2 = ConvBNAct(hidden, channels, kernel_size=3)
        self.shortcut = shortcut

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return x + out if self.shortcut else out


class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=1, expansion=0.5):
        super().__init__()
        hidden = max(1, int(out_channels * expansion))
        self.left = ConvBNAct(in_channels, hidden, kernel_size=1)
        self.right = ConvBNAct(in_channels, hidden, kernel_size=1)
        self.blocks = nn.Sequential(*[Bottleneck(hidden, expansion=1.0) for _ in range(num_blocks)])
        self.out = ConvBNAct(hidden * 2, out_channels, kernel_size=1)

    def forward(self, x):
        left = self.blocks(self.left(x))
        right = self.right(x)
        return self.out(torch.cat([left, right], dim=1))


class CSPStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=1):
        super().__init__()
        self.downsample = ConvBNAct(in_channels, out_channels, kernel_size=3, stride=2)
        self.csp = CSPBlock(out_channels, out_channels, num_blocks=num_blocks)

    def forward(self, x):
        return self.csp(self.downsample(x))


class TopDownFPN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        c2, c3, c4 = channels
        self.reduce4 = ConvBNAct(c4, c3, kernel_size=1)
        self.fuse3 = CSPBlock(c3 * 2, c3, num_blocks=1)
        self.reduce3 = ConvBNAct(c3, c2, kernel_size=1)
        self.fuse2 = CSPBlock(c2 * 2, c2, num_blocks=1)

    def forward(self, c2, c3, c4):
        p4 = self.reduce4(c4)
        p3 = self.fuse3(torch.cat([F.interpolate(p4, size=c3.shape[-2:], mode='nearest'), c3], dim=1))
        p3_reduced = self.reduce3(p3)
        p2 = self.fuse2(torch.cat([F.interpolate(p3_reduced, size=c2.shape[-2:], mode='nearest'), c2], dim=1))
        return p2, p3, c4


class BottomUpPAN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        c2, c3, c4 = channels
        self.down2 = ConvBNAct(c2, c3, kernel_size=3, stride=2)
        self.fuse3 = CSPBlock(c3 * 2, c3, num_blocks=1)
        self.down3 = ConvBNAct(c3, c4, kernel_size=3, stride=2)
        self.fuse4 = CSPBlock(c4 * 2, c4, num_blocks=1)

    def forward(self, p2, p3, p4):
        n3 = self.fuse3(torch.cat([self.down2(p2), p3], dim=1))
        n4 = self.fuse4(torch.cat([self.down3(n3), p4], dim=1))
        return p2, n3, n4


class DecoupledDenseHead(nn.Module):
    def __init__(self, in_channels, object_classes, geometry_channels, head_channels):
        super().__init__()
        self.object_tower = nn.Sequential(
            ConvBNAct(in_channels, head_channels, kernel_size=3),
            ConvBNAct(head_channels, head_channels, kernel_size=3),
        )
        self.geometry_tower = nn.Sequential(
            ConvBNAct(in_channels, head_channels, kernel_size=3),
            ConvBNAct(head_channels, head_channels, kernel_size=3),
        )
        self.object_pred = nn.Conv2d(head_channels, object_classes, kernel_size=1)
        self.geometry_pred = nn.Conv2d(head_channels, geometry_channels, kernel_size=1)

    def forward(self, x):
        object_logits = self.object_pred(self.object_tower(x))
        geometry = self.geometry_pred(self.geometry_tower(x))
        return torch.cat([object_logits, geometry], dim=1)


class CSPPAFPNNet(nn.Module):
    """CSP backbone + PAN/FPN neck for dense barcode heatmap, geometry and ROI quality."""

    def __init__(
            self,
            in_channels,
            out_channels=None,
            object_classes=3,
            geometry_channels=6,
            base_channels=24,
            depth=1,
            head_channels=48,
            interpolation='nearest',
            **_):
        super().__init__()
        self.object_classes = int(object_classes)
        self.geometry_channels = int(geometry_channels)
        self.out_channels = int(out_channels or (self.object_classes + self.geometry_channels))
        self.feature_channels = int(head_channels)
        self.interpolation = interpolation

        c1 = int(base_channels)
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 8
        blocks = max(1, int(depth))

        self.stem = nn.Sequential(
            ConvBNAct(in_channels, c1, kernel_size=3, stride=2),
            ConvBNAct(c1, c1, kernel_size=3),
        )
        self.stage2 = CSPStage(c1, c2, num_blocks=blocks)
        self.stage3 = CSPStage(c2, c3, num_blocks=blocks + 1)
        self.stage4 = CSPStage(c3, c4, num_blocks=blocks + 1)

        self.fpn = TopDownFPN((c2, c3, c4))
        self.pan = BottomUpPAN((c2, c3, c4))
        self.project2 = ConvBNAct(c2, head_channels, kernel_size=1)
        self.project3 = ConvBNAct(c3, head_channels, kernel_size=1)
        self.project4 = ConvBNAct(c4, head_channels, kernel_size=1)
        self.feature_refine = nn.Sequential(
            ConvBNAct(head_channels * 3, head_channels, kernel_size=3),
            ConvBNAct(head_channels, head_channels, kernel_size=3),
        )
        self.head = DecoupledDenseHead(
            in_channels=head_channels,
            object_classes=self.object_classes,
            geometry_channels=self.geometry_channels,
            head_channels=head_channels)

        if self.out_channels != self.object_classes + self.geometry_channels:
            raise ValueError('CSPPAFPNNet out_channels must equal object_classes + geometry_channels')

    def forward(self, x, return_features=False):
        input_size = x.shape[-2:]
        c1 = self.stem(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        p2, p3, p4 = self.fpn(c2, c3, c4)
        n2, n3, n4 = self.pan(p2, p3, p4)

        f2 = self.project2(n2)
        f3 = F.interpolate(self.project3(n3), size=f2.shape[-2:], mode=self.interpolation)
        f4 = F.interpolate(self.project4(n4), size=f2.shape[-2:], mode=self.interpolation)
        feature = self.feature_refine(torch.cat([f2, f3, f4], dim=1))
        feature = F.interpolate(feature, size=input_size, mode=self.interpolation)
        dense_output = self.head(feature)

        if return_features:
            return dense_output, feature
        return dense_output
