import torch
from torch import nn


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

        self.dbcb = []
        self.n = len(convs)

        self.dbcb.append(DilatedBranchConvBlock(convs[0],
                                                in_channels[1],
                                                h_channels[0]))

        for i in range(1, self.n):
            self.dbcb.append(DilatedBranchConvBlock(convs[i],
                                                    h_channels[i - 1],
                                                    h_channels[i]))

        nf1 = sizes[0] * sizes[1] * h_channels[-1]
        nf2 = int(sizes[0] * sizes[1] * h_channels[-1] / 4.0)
        nf3 = int(sizes[0] * sizes[1] * h_channels[-1] / 16.0)
        nf4 = int(sizes[0] * sizes[1] * h_channels[-1] / 64.0)

        self.fc = nn.Linear(nf1 + nf2 + nf3 + nf4, out_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        sizes = x.shape
        x = self.mrb(x)

        for i in range(self.n):
            x = self.dbcb[i](x)

        x0 = x[0].view(sizes[0], -1)
        x1 = x[1].view(sizes[0], -1)
        x2 = x[2].view(sizes[0], -1)
        x3 = x[3].view(sizes[0], -1)

        out = torch.cat((x0, x1, x2, x3), dim=1)
        out = self.softmax(self.fc(out))

        return out


class MultiResolutionBlock(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()

        # args = [(stride, padding, dilation),
        #         (...),
        #         (stride, padding, dilation)
        #         ]
        args = [(1, 4, 4),
                (2, 3, 3),
                (4, 2, 2),
                (8, 1, 1)
                ]

        self.convs = []
        for el in args:
            s, p, d = el[0], el[1], el[2]
            self.convs.append(nn.Conv2d(in_channels=inc,
                                        out_channels=outc,
                                        kernel_size=(3, 3),
                                        stride=s,
                                        padding=p,
                                        dilation=d))
        self.act = nn.ReLU()

    def forward(self, x):
        fm1 = self.act(self.convs[0](x))
        fm2 = self.act(self.convs[1](x))
        fm3 = self.act(self.convs[2](x))
        fm4 = self.act(self.convs[3](x))

        return [fm1, fm2, fm3, fm4]


class DilatedBranchConvSingle(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()

        # args = [(stride, padding, dilation, output_padding),
        #         (...),
        #         (stride, padding, dilation, output_padding)
        #         ]
        args = [(1, 4, 4, -1),
                (2, 3, 3, 1),
                (1, 3, 3, -1),
                (2, 4, 4, -1),
                (1, 2, 2, -1),
                (2, 1, 1, 1),
                (1, 1, 1, -1),
                (2, 2, 2, -1),
                ]

        self.convs = []
        for el in args:
            s, p, d, op = el[0], el[1], el[2], el[3]
            if op == -1:
                self.convs.append(nn.Conv2d(in_channels=inc,
                                            out_channels=outc,
                                            kernel_size=(3, 3),
                                            stride=s,
                                            padding=p,
                                            dilation=d))
            else:
                self.convs.append(nn.ConvTranspose2d(in_channels=inc,
                                                     out_channels=outc,
                                                     kernel_size=(3, 3),
                                                     stride=s,
                                                     padding=p,
                                                     dilation=d,
                                                     output_padding=op))

        self.bottlenecks = []
        for _ in range(4):
            self.bottlenecks.append(nn.Conv2d(in_channels=int(outc * 2),
                                              out_channels=outc,
                                              kernel_size=(1, 1),
                                              stride=1,
                                              padding=0,
                                              dilation=1))

        self.act = nn.ReLU()

    def forward(self, fm):
        h1_sr = self.act(self.convs[0](fm[0]))
        h1_dr = self.act(self.convs[1](fm[1]))

        h2_sr = self.act(self.convs[2](fm[1]))
        h2_dr = self.act(self.convs[3](fm[0]))

        h3_sr = self.act(self.convs[4](fm[2]))
        h3_dr = self.act(self.convs[5](fm[3]))

        h4_sr = self.act(self.convs[6](fm[3]))
        h4_dr = self.act(self.convs[7](fm[2]))

        h1 = self.act(self.bottlenecks[0](torch.cat((h1_sr, h1_dr), dim=1)))
        h2 = self.act(self.bottlenecks[1](torch.cat((h2_sr, h2_dr), dim=1)))
        h3 = self.act(self.bottlenecks[2](torch.cat((h3_sr, h3_dr), dim=1)))
        h4 = self.act(self.bottlenecks[3](torch.cat((h4_sr, h4_dr), dim=1)))

        return [h1, h2, h3, h4]


class DilatedBranchConvSingleLast(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()

        # args = [(stride, padding, dilation, output_padding),
        #         (...),
        #         (stride, padding, dilation, output_padding)
        #         ]
        args = [(1, 4, 4, -1),
                (2, 4, 4, -1),
                (4, 4, 4, -1),
                (8, 4, 4, -1),
                (2, 3, 3, 1),
                (1, 3, 3, -1),
                (2, 3, 3, -1),
                (4, 3, 3, -1),
                (4, 2, 2, 3),
                (2, 2, 2, 1),
                (1, 2, 2, -1),
                (2, 2, 2, -1),
                (8, 1, 1, 7),
                (4, 1, 1, 3),
                (2, 1, 1, 1),
                (1, 1, 1, -1),
                ]

        self.convs = []
        for el in args:
            s, p, d, op = el[0], el[1], el[2], el[3]
            if op == -1:
                self.convs.append(nn.Conv2d(in_channels=inc,
                                            out_channels=outc,
                                            kernel_size=(3, 3),
                                            stride=s,
                                            padding=p,
                                            dilation=d))
            else:
                self.convs.append(nn.ConvTranspose2d(in_channels=inc,
                                                     out_channels=outc,
                                                     kernel_size=(3, 3),
                                                     stride=s,
                                                     padding=p,
                                                     dilation=d,
                                                     output_padding=op))

        self.bottlenecks = []
        for _ in range(4):
            self.bottlenecks.append(nn.Conv2d(in_channels=int(outc * 4),
                                              out_channels=outc,
                                              kernel_size=(1, 1),
                                              stride=1,
                                              padding=0,
                                              dilation=1))

        self.act = nn.ReLU()

    def forward(self, fm):
        h1i = [self.act(self.convs[i](fm[0])) for i in range(0, 4)]
        h2i = [self.act(self.convs[i](fm[1])) for i in range(4, 8)]
        h3i = [self.act(self.convs[i](fm[2])) for i in range(8, 12)]
        h4i = [self.act(self.convs[i](fm[3])) for i in range(12, 16)]

        h1 = self.act(self.bottlenecks[0](
            torch.cat((h1i[0], h2i[0], h3i[0], h4i[0]), dim=1)))
        h2 = self.act(self.bottlenecks[1](
            torch.cat((h1i[1], h2i[1], h3i[1], h4i[1]), dim=1)))
        h3 = self.act(self.bottlenecks[2](
            torch.cat((h1i[2], h2i[2], h3i[2], h4i[2]), dim=1)))
        h4 = self.act(self.bottlenecks[3](
            torch.cat((h1i[3], h2i[3], h3i[3], h4i[3]), dim=1)))

        return [h1, h2, h3, h4]


class DilatedBranchConvBlock(nn.Module):
    def __init__(self, n_convs, inc, outc):
        super().__init__()

        self.n_convs = n_convs
        self.dbcs = []
        self.dbcs.append(DilatedBranchConvSingle(inc, outc))

        for _ in range(n_convs - 2):
            self.dbcs.append(DilatedBranchConvSingle(outc, outc))

        self.dbcs.append(DilatedBranchConvSingleLast(outc, outc))

    def forward(self, dbcs):
        for i in range(self.n_convs):
            dbcs = self.dbcs[i](dbcs)
        return dbcs
