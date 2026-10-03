"""
Attention U-Net variant of unet_refine.py -- identical overall shape (3-stage
encoder/decoder, same channel widths 32/64/128/256, same output-level
residual-onto-warp trick), with exactly ONE change from the plain U-Net:
every skip connection passes through an Attention Gate (Oktay et al., 2018)
before being concatenated into the decoder, instead of being concatenated
raw.

Where does this differ from resunet_refine.py's approach? ResUNet added
MORE COMPUTATION (residual connections inside every conv block) -- and it
made results slightly worse, suggesting this task isn't capacity-starved.
Attention gates test a different hypothesis entirely: not "more compute,"
but "smarter use of the SAME encoder features" -- the decoder learns to
weight which spatial regions of each skip connection actually matter
(e.g. cloud edges) versus which are irrelevant (e.g. flat clear sky),
before combining them. Same input (6ch), same output (2ch), same training
setup -- only the skip-connection handling differs.
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


class AttentionGate(nn.Module):
    """
    Standard additive attention gate (Oktay et al., "Attention U-Net", 2018).

    g: gating signal from the decoder (coarser, more semantic context) --
       already upsampled to match x's spatial size.
    x: the encoder skip-connection features being gated.

    Produces a per-pixel attention coefficient in [0,1] and multiplies x by
    it -- the network learns to suppress irrelevant skip-connection regions
    (e.g. flat clear-sky background) and emphasize relevant ones (e.g.
    cloud edges/texture) BEFORE the decoder ever sees them.
    """

    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        attention = self.psi(self.relu(g1 + x1))
        return x * attention


class AttentionUNetRefine(nn.Module):
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
        self.att3 = AttentionGate(gate_channels=128, skip_channels=128, inter_channels=64)
        self.dec3 = conv_block(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att2 = AttentionGate(gate_channels=64, skip_channels=64, inter_channels=32)
        self.dec2 = conv_block(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.att1 = AttentionGate(gate_channels=32, skip_channels=32, inter_channels=16)
        self.dec1 = conv_block(64, 32)

        self.out_conv = nn.Conv2d(32, final_out, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        e3_gated = self.att3(g=d3, x=e3)
        d3 = self.dec3(torch.cat([d3, e3_gated], dim=1))

        d2 = self.up2(d3)
        e2_gated = self.att2(g=d2, x=e2)
        d2 = self.dec2(torch.cat([d2, e2_gated], dim=1))

        d1 = self.up1(d2)
        e1_gated = self.att1(g=d1, x=e1)
        d1 = self.dec1(torch.cat([d1, e1_gated], dim=1))

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
