from collections import OrderedDict
import torch
from torch import nn

from .mrs import MultiResolutionBlock
from .dbcb import DilatedBranchConvBlock


__all__ = ['DMRNet']


class DMRNet(nn.Module):
    def __init__(self,
                 in_channels=[3, 16],
                 convs=[2, 4, 4, 6],
                 h_channels=[32, 32, 64, 64],
                 out_channels=5):
        super().__init__()

        if len(in_channels) != 2:
            raise RuntimeError('in_channels must be a list of two elements')

        if len(convs) < 2:
            raise RuntimeError('convs must contain at least two elements')

        if len(convs) != len(h_channels):
            raise RuntimeError('h_channels must have same shape as convs')

        self.mrb = MultiResolutionBlock(inc=in_channels[0],
                                        outc=in_channels[1])

        blocks = []
        self.n = len(convs)

        blocks.append((f'block_1', DilatedBranchConvBlock(convs[0],
                                                          in_channels[1],
                                                          h_channels[0])))

        for i in range(1, self.n):
            blocks.append((f'block_{i + 1}',
                           DilatedBranchConvBlock(convs[i],
                                                  h_channels[i - 1],
                                                  h_channels[i])))

        self.blocks = nn.Sequential(OrderedDict(blocks))

        self.avgpool1 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.avgpool3 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.avgpool4 = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(int(h_channels[-1] * 4), out_channels)

    def forward(self, x):
        shapes = x.shape

        x = self.mrb(x)
        blocks = self.blocks(x)

        x0 = self.avgpool1(blocks[0])
        x1 = self.avgpool2(blocks[1])
        x2 = self.avgpool3(blocks[2])
        x3 = self.avgpool4(blocks[3])

        out = torch.cat((x0, x1, x2, x3), dim=1)
        out = self.fc(out.view(shapes[0], -1))

        return out
