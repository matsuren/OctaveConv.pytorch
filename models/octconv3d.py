from functools import partial

import torch.nn as nn
import torch.nn.functional as F


class OctConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 alpha_in=0.25, alpha_out=0.25, type='normal'):
        super(OctConv3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.type = type
        hf_ch_in = int(in_channels * (1 - alpha_in))
        hf_ch_out = int(out_channels * (1 - alpha_out))
        lf_ch_in = in_channels - hf_ch_in
        lf_ch_out = out_channels - hf_ch_out

        if type == 'first':
            if stride == 2:
                self.downsample = nn.AvgPool3d(kernel_size=2, stride=stride)
            self.convh = nn.Conv3d(
                in_channels, hf_ch_out,
                kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation,
            )
            self.avg_pool = nn.AvgPool3d(kernel_size=2, stride=2)
            self.convl = nn.Conv3d(
                in_channels, lf_ch_out,
                kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, 
            )
        elif type == 'last':
            if stride == 2:
                self.downsample = nn.AvgPool3d(kernel_size=2, stride=stride)
            self.convh = nn.Conv3d(hf_ch_in, out_channels, kernel_size=kernel_size, padding=padding,dilation=dilation)
            self.convl = nn.Conv3d(lf_ch_in, out_channels, kernel_size=kernel_size, padding=padding,dilation=dilation)
            self.upsample = partial(F.interpolate, scale_factor=2, mode="nearest")
        else:
            if stride == 2:
                self.downsample = nn.AvgPool3d(kernel_size=2, stride=stride)

            self.L2L = nn.Conv3d(
                lf_ch_in, lf_ch_out,
                kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation
            )
            self.L2H = nn.Conv3d(
                lf_ch_in, hf_ch_out,
                kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation
            )
            self.H2L = nn.Conv3d(
                hf_ch_in, lf_ch_out,
                kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation
            )
            self.H2H = nn.Conv3d(
                hf_ch_in, hf_ch_out,
                kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation
            )
            self.upsample = partial(F.interpolate, scale_factor=2, mode="nearest")
            self.avg_pool = partial(F.avg_pool3d, kernel_size=2, stride=2)

    def forward(self, x):
        if self.type == 'first':
            if self.stride == 2:
                x = self.downsample(x)

            hf = self.convh(x)
            lf = self.avg_pool(x)
            lf = self.convl(lf)

            return hf, lf
        elif self.type == 'last':
            hf, lf = x
            if self.stride == 2:
                hf = self.downsample(hf)
                return self.convh(hf) + self.convl(lf)
            else:
                return self.convh(hf) + self.convl(self.upsample(lf))
        else:
            hf, lf = x
            if self.stride == 2:
                hf = self.downsample(hf)
                return self.H2H(hf) + self.L2H(lf), \
                       self.L2L(F.avg_pool3d(lf, kernel_size=2, stride=2)) + self.H2L(self.avg_pool(hf))
            else:
                return self.H2H(hf) + self.upsample(self.L2H(lf)), self.L2L(lf) + self.H2L(self.avg_pool(hf))

