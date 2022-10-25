# DMRNet

This is my first CNN architecture. I realised a main idea - multi-resolution branches with different dilation number, but it isn't working in production now.

![Architecture](/images/dmrnet.png "DMRNet")

## Annotation
- This model isn't working good, training speed of this model is more than 30 times slower than resnet18 with same number of trainable parametrs.
- I used 4 branches with same resolution in DMRNet, each branch has own dilation number.
- Convolutions with same in_channels and out_channels have residual connections.
- DMRNet consists of DilatedBranchConvBlock, each block ends with cross-branch convolutions. Intermediate convolutions in block also have cross-branch connection branch1 - branch2 and branch3 - branch4.
- Cross-branch-connections have same dilation as start connection branch.
- ConvTranspose2D used for increasing the resolution.

## Model configuration
```python
from model import DMRNet

# [input images channels number, mrs output channels number]
in_channels = [3, 32]

# len(convs) = number of blocks
# each value in convs is a number of convolutions in block
convs = [2, 4, 4, 6]

# len(h_channels) = number of blocks
# each value in h_channels is a number of output channels for convolutions
h_channels = [32, 64, 64, 128]

# number of output channels
out_channels = 10

model = DMRNet(in_channels=in_channels,
               convs=convs,
               h_channels=h_channels,
               out_channels=out_channels)
```

## Project structure
- model/ - architecture.
- train.py - training algorithm.
- train.yaml - config file for training algorith and model configuration.
