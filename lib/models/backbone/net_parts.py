# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = nn.Sequential(
            double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.conv(x)
        return x


class Swing_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Swing_down, self).__init__()
        self.dconv = double_conv(out_ch, out_ch)
        self.down = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)

            
    def forward(self, x):
        x = self.down(x)
        x = self.dconv(x)
        return x


class res_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(res_block, self).__init__()
        self.res1 = nn.Sequential(ChannelAttentionModule(in_ch, reduction_ratio=16),
                                  double_conv(in_ch, int(out_ch/2)))
        self.res2 = nn.Sequential(SpatialAttention(),
                                  double_conv(in_ch, int(out_ch/2)))
        self.downsample = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(out_ch))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        xt = self.downsample(x)
        x1 = self.res1(x)
        x2 = self.res2(x)
        xo = torch.cat((x1, x2), 1)
        return self.relu(xo + xt)


class Swing_up(nn.Module):
    def __init__(self, mid_ch, out_ch):
        super(Swing_up, self).__init__()
        self.up =  nn.ConvTranspose2d(mid_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.dconv = double_conv(out_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.dconv(x)
        return x


# class Unet_block(nn.Module):
#     def __init__(self, in_ch, mid_ch, out_ch):
#         super(Unet_block, self).__init__()
#         self.down = Swing_down(in_ch, mid_ch)
#         self.up = Swing_up(mid_ch, out_ch)
#         self.res = res_block(mid_ch, mid_ch)
#     def forward(self, x):
#         x = self.down(x)
#         x = self.res(x)
#         x = self.up(x)
#         return x

class Unet_block(nn.Module):
    def __init__(self, in_ch, w):
        super(Unet_block, self).__init__()
        self.infer = nn.Sequential()
        for i in range(w):
            self.infer.append(Swing_down(in_ch*(i+1), in_ch*(i+2)))
        self.infer.append(res_block(in_ch*(w+1),in_ch*(w+1)))
        for i in range(w):
            self.infer.append(Swing_up(in_ch*(w+1-i), in_ch*(w-i)))
    def forward(self, x):
        x = self.infer(x)
        return x



class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 == 1, "Kernel size must be odd."
        self.padding = 3 if self.kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=self.padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        atten = torch.cat([avg_out, max_out], dim=1)
        atten = self.conv(atten)
        atten = self.sigmoid(atten)
        scaled_attention = atten * x
        return scaled_attention


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
