
import torch
import torch.nn as nn
import torch.nn.functional as F
from .net_parts import inconv, Swing_down, Swing_up, double_conv, Unet_block


class Swingnet(nn.Module):
    def __init__(self, channels_list, in_c, w, n_classes):
        super(Swingnet, self).__init__()
        self.n_classes = n_classes
        self.swnet = nn.Sequential()
        self.swnet.append(inconv(in_c, 32))
        for inc  in channels_list:
            self.swnet.append(Unet_block(inc, w))

        self.swnet.append(double_conv(channels_list[-1], n_classes))
        

    def forward(self, x):
        a = self.swnet(x)
        return a

def swing_net(config):
    l = config['swing_longth']  
    w = config['swing_width']  
    x = config['in_channels']  
    cls = config['N_CLASSES']
    channel_list = []

    for i in range(l):
        channel_list.append(32)
    return Swingnet(channel_list, x, w, n_classes=cls)


