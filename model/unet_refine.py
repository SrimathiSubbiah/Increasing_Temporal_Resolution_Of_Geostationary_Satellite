"""
Small U-Net that refines a TV-L1-warped intermediate frame.

Input: concat of [warped_frame, frame0, frame2] = 2 channels x 3 = 6 channels
Output: refined frame1 prediction, 2 channels (Band13, Band9)

This is the scoped-down stand-in for WR-Net's non-local multi-temporal fusion
layer -- a plain concatenation instead of attention-based fusion. Document
this explicitly as a scoping decision in your report: given more time, the
non-local fusion (query=warped frame, key/value=frame0/frame2) would likely
improve results further, but a concat-U-Net isolates whether refinement
itself helps before adding fusion complexity.
"""

import torch
import torch.nn as nn


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNetRefine(nn.Module):
    def __init__(self, in_channels=6, out_channels=2, predict_uncertainty=False, residual=True):
        super().__init__()
        self.predict_uncertainty = predict_uncertainty
        self.residual = residual
        self.out_channels = out_channels
        final_out = out_channels * 2 if predict_uncertainty else out_channels

        self.enc1 = conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = conv_block(64, 32)

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
            # x = concat([warped, frame0, frame2]) along channel dim -- the first
            # out_channels channels are the warped frame this network refines.
            # Predicting a delta on top of it (instead of the pixel values from
            # scratch) means the identity mapping is free, so the network only
            # has to learn to do better than what it's handed.
            warped = x[:, : self.out_channels]
            if self.predict_uncertainty:
                mean, log_var = torch.chunk(out, 2, dim=1)
                return warped + mean, log_var
            return warped + out

        if self.predict_uncertainty:
            mean, log_var = torch.chunk(out, 2, dim=1)
            return mean, log_var
        return out