# DMRNet

This is my first CNN architecture. I implemented one main idea - multi-resolution branches with different dilation numbers, but it doesn't work in production now.

![Architecture](/images/dmrnet.png "DMRNet")

## Annotation
- This model doesn't work well, the training speed of this model is more than 30 times slower than resnet18 with the same number of trainable parameters.
- I used 4 branches with same resolution in DMRNet, each branch has own dilation number.
- Convolutions with same in_channels and out_channels have residual connections.
- DMRNet consists of DilatedBranchConvBlock, each block ends with cross-branch convolutions. Intermediate convolutions in the block also have cross-branch connections branch1 - branch2 and branch3 - branch4.
- Cross-branch connections have the same dilation as the start branch connection.
- ConvTranspose2D used to increase resolution.

## Model configuration
```python
from model import DMRNet

# [input images channels number, mrs output channels number]
in_channels = [3, 32]

# len(convs) = number of blocks
# each value in convs is the number of convolutions in the block
convs = [2, 4, 4, 6]

# len(h_channels) = number of blocks
# each value in h_channels is the number of output channels for convolutions
h_channels = [32, 64, 64, 128]

# number of output channels
out_channels = 10

model = DMRNet(in_channels=in_channels,
               convs=convs,
               h_channels=h_channels,
               out_channels=out_channels)
```

## Project structure
- model - architecture.
- train.py - training algorithm.
- train.yaml - configuration file for training algorithm and model configuration.
