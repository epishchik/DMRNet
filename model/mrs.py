from torch import nn


__all__ = ['MultiResolutionBlock']


class MultiResolutionBlock(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()

        self.branch_1 = nn.Conv2d(in_channels=inc,
                                  out_channels=outc,
                                  kernel_size=(3, 3),
                                  stride=1,
                                  padding=4,
                                  dilation=4)

        self.branch_2 = nn.Conv2d(in_channels=inc,
                                  out_channels=outc,
                                  kernel_size=(3, 3),
                                  stride=2,
                                  padding=3,
                                  dilation=3)

        self.branch_3 = nn.Conv2d(in_channels=inc,
                                  out_channels=outc,
                                  kernel_size=(3, 3),
                                  stride=4,
                                  padding=2,
                                  dilation=2)

        self.branch_4 = nn.Conv2d(in_channels=inc,
                                  out_channels=outc,
                                  kernel_size=(3, 3),
                                  stride=8,
                                  padding=1,
                                  dilation=1)

        self.act = nn.ReLU()

    def forward(self, x):
        fm1 = self.act(self.branch_1(x))
        fm2 = self.act(self.branch_2(x))
        fm3 = self.act(self.branch_3(x))
        fm4 = self.act(self.branch_4(x))

        return [fm1, fm2, fm3, fm4]
