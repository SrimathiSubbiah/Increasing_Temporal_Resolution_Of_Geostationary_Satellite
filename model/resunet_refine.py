"""
ResUNet variant of unet_refine.py -- identical overall shape (3-stage
encoder/decoder, same channel widths 32/64/128/256, same skip connections,
same output-level residual-onto-warp trick), with exactly ONE change:
every conv_block now has an internal residual connection (input added back
to the block's output before the final activation), not just plain
Conv-BN-ReLU x2 in series.

This isolates a single variable against unet_refine.py: does residual
learning help more when applied throughout the network, not just once at
the very output? Same input (6ch: warped+frame0+frame2), same output (2ch),
same training script, same data -- the only difference is this file.
"""

import torch
import torch.nn as nn


class ResConvBlock(nn.Module):
    """Conv-BN-ReLU x2, with the block's input added back before the final
    activation. A 1x1 conv projects the input when in_ch != out_ch, since
    the residual add requires matching channel counts."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.proj = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.proj(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class ResUNetRefine(nn.Module):
    def __init__(self, in_channels=6, out_channels=2, predict_uncertainty=False, residual=True):
        super().__init__()
        self.predict_uncertainty = predict_uncertainty
        self.residual = residual
        self.out_channels = out_channels
        final_out = out_channels * 2 if predict_uncertainty else out_channels

        self.enc1 = ResConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ResConvBlock(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ResConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ResConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ResConvBlock(64, 32)

        self.out_conv = nn.Conv2d(32, final_out, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.out_conv(d1)

        if self.residual:
            warped = x[:, : self.out_channels]
            if self.predict_uncertainty:
                mean, log_var = torch.chunk(out, 2, dim=1)
                return warped + mean, log_var
            return warped + out

        if self.predict_uncertainty:
            mean, log_var = torch.chunk(out, 2, dim=1)
            return mean, log_var
        return out
