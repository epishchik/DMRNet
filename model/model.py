from collections import OrderedDict
import torch
from torch import nn

from .mrs import MultiResolutionBlock
from .dbcb import DilatedBranchConvBlock


__all__ = ['DMRNet']


class DMRNet(nn.Module):
    def __init__(self,
                 sizes=(256, 256),
                 in_channels=[3, 16],
                 convs=[2, 4, 4, 6],
                 h_channels=[32, 32, 64, 64],
                 out_channels=5):
        super().__init__()

        if len(sizes) != 2:
            raise RuntimeError('sizes must be a tuple of two elements')

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

        nf1 = sizes[0] * sizes[1] * h_channels[-1]
        nf2 = int(sizes[0] * sizes[1] * h_channels[-1] / 4.0)
        nf3 = int(sizes[0] * sizes[1] * h_channels[-1] / 16.0)
        nf4 = int(sizes[0] * sizes[1] * h_channels[-1] / 64.0)

        self.blocks = nn.Sequential(OrderedDict(blocks))
        self.fc = nn.Linear(nf1 + nf2 + nf3 + nf4, out_channels)

    def forward(self, x):
        sizes = x.shape

        x = self.mrb(x)
        blocks = self.blocks(x)

        x0 = blocks[0].view(sizes[0], -1)
        x1 = blocks[1].view(sizes[0], -1)
        x2 = blocks[2].view(sizes[0], -1)
        x3 = blocks[3].view(sizes[0], -1)

        out = torch.cat((x0, x1, x2, x3), dim=1)
        out = self.fc(out)

        return out
