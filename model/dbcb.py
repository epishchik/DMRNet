from collections import OrderedDict
import torch
from torch import nn


__all__ = ['DilatedBranchConvBlock']


class DilatedBranchConvBlock(nn.Module):
    def __init__(self, n_convs, inc, outc):
        super().__init__()

        self.n_convs = n_convs
        block_convs = []
        block_convs.append(
            ('block_conv_1', DilatedBranchConvSingle(inc, outc)))

        for i in range(n_convs - 2):
            block_convs.append(
                (f'block_conv_{i + 2}', DilatedBranchConvSingle(outc, outc)))

        block_convs.append((f'block_conv_{n_convs}',
                            DilatedBranchConvSingleLast(outc, outc)))

        self.block_convs = nn.Sequential(OrderedDict(block_convs))

    def forward(self, fm):
        block_convs = self.block_convs(fm)
        return block_convs


class DilatedBranchConvSingle(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.identity = inc == outc

        self.branch_1_same_resolution = nn.Conv2d(in_channels=inc,
                                                  out_channels=outc,
                                                  kernel_size=(3, 3),
                                                  stride=1,
                                                  padding=4,
                                                  dilation=4)

        self.branch_1_different_resolution = nn.ConvTranspose2d(
            in_channels=inc,
            out_channels=outc,
            kernel_size=(3, 3),
            stride=2,
            padding=3,
            dilation=3,
            output_padding=1)

        self.branch_2_same_resolution = nn.Conv2d(in_channels=inc,
                                                  out_channels=outc,
                                                  kernel_size=(3, 3),
                                                  stride=1,
                                                  padding=3,
                                                  dilation=3)

        self.branch_2_different_resolution = nn.Conv2d(in_channels=inc,
                                                       out_channels=outc,
                                                       kernel_size=(3, 3),
                                                       stride=2,
                                                       padding=4,
                                                       dilation=4)

        self.branch_3_same_resolution = nn.Conv2d(in_channels=inc,
                                                  out_channels=outc,
                                                  kernel_size=(3, 3),
                                                  stride=1,
                                                  padding=2,
                                                  dilation=2)

        self.branch_3_different_resolution = nn.ConvTranspose2d(
            in_channels=inc,
            out_channels=outc,
            kernel_size=(3, 3),
            stride=2,
            padding=1,
            dilation=1,
            output_padding=1)

        self.branch_4_same_resolution = nn.Conv2d(in_channels=inc,
                                                  out_channels=outc,
                                                  kernel_size=(3, 3),
                                                  stride=1,
                                                  padding=1,
                                                  dilation=1)

        self.branch_4_different_resolution = nn.Conv2d(in_channels=inc,
                                                       out_channels=outc,
                                                       kernel_size=(3, 3),
                                                       stride=2,
                                                       padding=2,
                                                       dilation=2)

        self.branch_1_bottleneck = nn.Conv2d(in_channels=int(outc * 2),
                                             out_channels=outc,
                                             kernel_size=(1, 1),
                                             stride=1,
                                             padding=0,
                                             dilation=1)

        self.branch_2_bottleneck = nn.Conv2d(in_channels=int(outc * 2),
                                             out_channels=outc,
                                             kernel_size=(1, 1),
                                             stride=1,
                                             padding=0,
                                             dilation=1)

        self.branch_3_bottleneck = nn.Conv2d(in_channels=int(outc * 2),
                                             out_channels=outc,
                                             kernel_size=(1, 1),
                                             stride=1,
                                             padding=0,
                                             dilation=1)

        self.branch_4_bottleneck = nn.Conv2d(in_channels=int(outc * 2),
                                             out_channels=outc,
                                             kernel_size=(1, 1),
                                             stride=1,
                                             padding=0,
                                             dilation=1)

        self.act = nn.ReLU()

    def forward(self, fm):
        h1_sr = self.act(self.branch_1_same_resolution(fm[0]))
        h1_dr = self.act(self.branch_1_different_resolution(fm[1]))

        h2_sr = self.act(self.branch_2_same_resolution(fm[1]))
        h2_dr = self.act(self.branch_2_different_resolution(fm[0]))

        h3_sr = self.act(self.branch_3_same_resolution(fm[2]))
        h3_dr = self.act(self.branch_3_different_resolution(fm[3]))

        h4_sr = self.act(self.branch_4_same_resolution(fm[3]))
        h4_dr = self.act(self.branch_4_different_resolution(fm[2]))

        h1 = torch.cat((h1_sr, h1_dr), dim=1)
        h2 = torch.cat((h2_sr, h2_dr), dim=1)
        h3 = torch.cat((h3_sr, h3_dr), dim=1)
        h4 = torch.cat((h4_sr, h4_dr), dim=1)

        h1 = self.act(self.branch_1_bottleneck(h1))
        h2 = self.act(self.branch_2_bottleneck(h2))
        h3 = self.act(self.branch_3_bottleneck(h3))
        h4 = self.act(self.branch_4_bottleneck(h4))

        if self.identity:
            h1 = h1 + fm[0]
            h2 = h2 + fm[1]
            h3 = h3 + fm[2]
            h4 = h4 + fm[3]

        return [h1, h2, h3, h4]


class DilatedBranchConvSingleLast(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.identity = inc == outc

        self.branch_1_to_branch_1 = nn.Conv2d(in_channels=inc,
                                              out_channels=outc,
                                              kernel_size=(3, 3),
                                              stride=1,
                                              padding=4,
                                              dilation=4)

        self.branch_1_to_branch_2 = nn.Conv2d(in_channels=inc,
                                              out_channels=outc,
                                              kernel_size=(3, 3),
                                              stride=2,
                                              padding=4,
                                              dilation=4)

        self.branch_1_to_branch_3 = nn.Conv2d(in_channels=inc,
                                              out_channels=outc,
                                              kernel_size=(3, 3),
                                              stride=4,
                                              padding=4,
                                              dilation=4)

        self.branch_1_to_branch_4 = nn.Conv2d(in_channels=inc,
                                              out_channels=outc,
                                              kernel_size=(3, 3),
                                              stride=8,
                                              padding=4,
                                              dilation=4)

        self.branch_2_to_branch_1 = nn.ConvTranspose2d(in_channels=inc,
                                                       out_channels=outc,
                                                       kernel_size=(3, 3),
                                                       stride=2,
                                                       padding=3,
                                                       dilation=3,
                                                       output_padding=1)

        self.branch_2_to_branch_2 = nn.Conv2d(in_channels=inc,
                                              out_channels=outc,
                                              kernel_size=(3, 3),
                                              stride=1,
                                              padding=3,
                                              dilation=3)

        self.branch_2_to_branch_3 = nn.Conv2d(in_channels=inc,
                                              out_channels=outc,
                                              kernel_size=(3, 3),
                                              stride=2,
                                              padding=3,
                                              dilation=3)

        self.branch_2_to_branch_4 = nn.Conv2d(in_channels=inc,
                                              out_channels=outc,
                                              kernel_size=(3, 3),
                                              stride=4,
                                              padding=3,
                                              dilation=3)

        self.branch_3_to_branch_1 = nn.ConvTranspose2d(in_channels=inc,
                                                       out_channels=outc,
                                                       kernel_size=(3, 3),
                                                       stride=4,
                                                       padding=2,
                                                       dilation=2,
                                                       output_padding=3)

        self.branch_3_to_branch_2 = nn.ConvTranspose2d(in_channels=inc,
                                                       out_channels=outc,
                                                       kernel_size=(3, 3),
                                                       stride=2,
                                                       padding=2,
                                                       dilation=2,
                                                       output_padding=1)

        self.branch_3_to_branch_3 = nn.Conv2d(in_channels=inc,
                                              out_channels=outc,
                                              kernel_size=(3, 3),
                                              stride=1,
                                              padding=2,
                                              dilation=2)

        self.branch_3_to_branch_4 = nn.Conv2d(in_channels=inc,
                                              out_channels=outc,
                                              kernel_size=(3, 3),
                                              stride=2,
                                              padding=2,
                                              dilation=2)

        self.branch_4_to_branch_1 = nn.ConvTranspose2d(in_channels=inc,
                                                       out_channels=outc,
                                                       kernel_size=(3, 3),
                                                       stride=8,
                                                       padding=1,
                                                       dilation=1,
                                                       output_padding=7)

        self.branch_4_to_branch_2 = nn.ConvTranspose2d(in_channels=inc,
                                                       out_channels=outc,
                                                       kernel_size=(3, 3),
                                                       stride=4,
                                                       padding=1,
                                                       dilation=1,
                                                       output_padding=3)

        self.branch_4_to_branch_3 = nn.ConvTranspose2d(in_channels=inc,
                                                       out_channels=outc,
                                                       kernel_size=(3, 3),
                                                       stride=2,
                                                       padding=1,
                                                       dilation=1,
                                                       output_padding=1)

        self.branch_4_to_branch_4 = nn.Conv2d(in_channels=inc,
                                              out_channels=outc,
                                              kernel_size=(3, 3),
                                              stride=1,
                                              padding=1,
                                              dilation=1)

        self.branch_1_bottleneck = nn.Conv2d(in_channels=int(outc * 4),
                                             out_channels=outc,
                                             kernel_size=(1, 1),
                                             stride=1,
                                             padding=0,
                                             dilation=1)

        self.branch_2_bottleneck = nn.Conv2d(in_channels=int(outc * 4),
                                             out_channels=outc,
                                             kernel_size=(1, 1),
                                             stride=1,
                                             padding=0,
                                             dilation=1)

        self.branch_3_bottleneck = nn.Conv2d(in_channels=int(outc * 4),
                                             out_channels=outc,
                                             kernel_size=(1, 1),
                                             stride=1,
                                             padding=0,
                                             dilation=1)

        self.branch_4_bottleneck = nn.Conv2d(in_channels=int(outc * 4),
                                             out_channels=outc,
                                             kernel_size=(1, 1),
                                             stride=1,
                                             padding=0,
                                             dilation=1)

        self.act = nn.ReLU()

    def forward(self, fm):
        h11 = self.act(self.branch_1_to_branch_1(fm[0]))
        h12 = self.act(self.branch_1_to_branch_2(fm[0]))
        h13 = self.act(self.branch_1_to_branch_3(fm[0]))
        h14 = self.act(self.branch_1_to_branch_4(fm[0]))

        h21 = self.act(self.branch_2_to_branch_1(fm[1]))
        h22 = self.act(self.branch_2_to_branch_2(fm[1]))
        h23 = self.act(self.branch_2_to_branch_3(fm[1]))
        h24 = self.act(self.branch_2_to_branch_4(fm[1]))

        h31 = self.act(self.branch_3_to_branch_1(fm[2]))
        h32 = self.act(self.branch_3_to_branch_2(fm[2]))
        h33 = self.act(self.branch_3_to_branch_3(fm[2]))
        h34 = self.act(self.branch_3_to_branch_4(fm[2]))

        h41 = self.act(self.branch_4_to_branch_1(fm[3]))
        h42 = self.act(self.branch_4_to_branch_2(fm[3]))
        h43 = self.act(self.branch_4_to_branch_3(fm[3]))
        h44 = self.act(self.branch_4_to_branch_4(fm[3]))

        h1 = torch.cat((h11, h21, h31, h41), dim=1)
        h2 = torch.cat((h12, h22, h32, h42), dim=1)
        h3 = torch.cat((h13, h23, h33, h43), dim=1)
        h4 = torch.cat((h14, h24, h34, h44), dim=1)

        h1 = self.act(self.branch_1_bottleneck(h1))
        h2 = self.act(self.branch_2_bottleneck(h2))
        h3 = self.act(self.branch_3_bottleneck(h3))
        h4 = self.act(self.branch_4_bottleneck(h4))

        if self.identity:
            h1 = h1 + fm[0]
            h2 = h2 + fm[1]
            h3 = h3 + fm[2]
            h4 = h4 + fm[3]

        return [h1, h2, h3, h4]
