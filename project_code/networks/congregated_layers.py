# Define more abstract U-Net classes
# Source: https://github.com/milesial/Pytorch-UNet/blob/6aa14cbbc445672d97190fec06d5568a0a004740/unet/unet_parts.py#L28

import torch.nn as nn

# Constructed according to typical behaviour in the U-Net
# where two convolutions with kernel=3 are stacked.
class DoubleConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        # In U-net we do two convolutions, but keep the channels the same
        # (so out_channels twice).
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            )
    
    def forward(self, x):
        return self.double_conv(x)

# Constructed according to typical behaviour in the U-Net
# where we downsample, immediately followed by doubling the
# channels by 2.
class Down(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        
    def forward(self, x):
        return self.maxpool_conv(x)

# Does the reverse of the Down class, since we want
# to reconstruct the image.
class Up(nn.Module):
    
    def __init__(self, in_channels, out_channels, bilinear):
        super().__init__()
        
        # If bilinear, use the normal convolution to reduce the number of channels.
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
            # Note that here we don't have the halve the in_channels to account for the concatenation.
            # This is because this is already compensated for in the UNet itself. You could see it as
            # the previous layer taking this already into account and therefore outputting half the
            # channel size for the upsampling method.
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            
            # Note that switching to in_channels again (as opposed to in_channels // 2),
            # makes sense because this includes the added features from the encoding part,
            # which doubles the in_channels size.
            self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Input is BCHW (batch, channel, height, width)
        # In order to account for the correct height and width size
        # when concatenating, we have to pad x1 to match x2.
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)
