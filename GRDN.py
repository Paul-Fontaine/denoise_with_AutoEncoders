import torch
import torch.nn as nn
import torch.nn.functional as F
from AutoEncoderBaseClass import AutoEncoderBaseClass


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers

        for i in range(num_layers):
            self.layers.append(nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1))

        # Final 1x1 convolution to merge features
        self.final_conv = nn.Conv2d(in_channels + num_layers * growth_rate, in_channels, kernel_size=1)

    def forward(self, x):
        inputs = [x]
        for i, layer in enumerate(self.layers):
            out = F.relu(layer(torch.cat(inputs, 1)))
            inputs.append(out)

        # Concatenate all intermediate features and the input
        concat_features = torch.cat(inputs, 1)
        # Final 1x1 convolution
        output = self.final_conv(concat_features)

        # Local residual learning
        return x + output


class GRDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, num_rdb_blocks):
        super(GRDB, self).__init__()
        self.rdb_blocks = nn.ModuleList()
        for _ in range(num_rdb_blocks):
            self.rdb_blocks.append(RDB(in_channels, growth_rate, num_layers))

        # Final 1x1 convolution to merge features
        self.final_conv = nn.Conv2d(in_channels * num_rdb_blocks, in_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        rdb_outputs = []

        for rdb in self.rdb_blocks:
            out = rdb(x)
            rdb_outputs.append(out)

        # Concatenate all RDB outputs
        concat_features = torch.cat(rdb_outputs, 1)

        # Final 1x1 convolution
        output = self.final_conv(concat_features)

        # Global residual learning
        return residual + output


class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        ch_att = self.channel_attention(x)
        x = x * ch_att
        sp_att = self.spatial_attention(x)
        x = x * sp_att
        return x


class GRDN(AutoEncoderBaseClass):
    def __init__(self, in_channels, out_channels, growth_rate, num_layers, num_rdb_blocks, num_grdb_blocks):
        super(GRDN, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_down = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

        self.grdb_blocks = nn.ModuleList()
        for _ in range(num_grdb_blocks):
            self.grdb_blocks.append(GRDB(out_channels, growth_rate, num_layers, num_rdb_blocks))

        self.conv_up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.final_conv = nn.Conv2d(out_channels, in_channels, kernel_size=3, padding=1)
        self.cbam = CBAM(out_channels)

    def forward(self, x):
        residual = x
        x = self.initial_conv(x)
        x = self.conv_down(x)

        for grdb in self.grdb_blocks:
            x = grdb(x)

        x = self.conv_up(x)
        x = self.cbam(x)
        x = self.final_conv(x)

        # Global residual learning
        x = residual + x

        return x
