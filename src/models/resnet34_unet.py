# Implement your ResNet34_UNet model here
import torch
from torch import nn
import torch.nn.init as init
import math

class res_block(nn.Module):
    def __init__(self, in_channel, out_channel, stride = 1):
        super(res_block, self).__init__()
        self.stride = stride
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (3, 3), stride = self.stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, (3, 3), stride = 1, padding=1),
            nn.BatchNorm2d(out_channel)
            )
        
        self.downsample = None
        if self.stride != 1 or in_channel != out_channel:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, (1, 1), stride = self.stride),
                nn.BatchNorm2d(out_channel)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        x = self.double_conv(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        x += identity
        x = self.relu(x)
        return x

class concat_upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(concat_upsample, self).__init__()
        self.up_convolution = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(2, 2), stride=2)
        init.normal_(self.up_convolution.weight, mean=0.0, std= math.sqrt(2/(9 * in_channel)))
        init.constant_(self.up_convolution.bias, 0)
        
        self.conv = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, (3, 3), padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, res_x, x):
        x = torch.cat([res_x, x], dim=1)
        x = self.up_convolution(x)
        x = self.conv(x)
        return x

class Resnet34_unet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Resnet34_unet, self).__init__()
        self.input_conv = nn.Sequential(
            #B * 3 * 256 * 256
            nn.Conv2d(in_channel, 64, (7, 7), stride=2, padding=3),
            #B * 64 * 128 * 128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), stride=2, padding=1)
            #B * 64 * 64 * 64
        )
        self.res_1 = self.make_layer(64, 64, 3, stride = 1)
        # first layer don't need downsample
        #B * 64 * 64 * 64
        self.res_2 = self.make_layer(64, 128, 4, stride = 2)
        #B * 128 * 32 * 32
        self.res_3 = self.make_layer(128, 256, 6, stride = 2)
        #B * 256 * 16 * 16
        self.res_4 = self.make_layer(256, 512, 3, stride = 2)
        #B * 512 * 8 * 8
        self.res_5 = self.make_layer(512, 256, 1, stride = 1) 
        # B * 256 * 8 * 8 
        #+B * 512 * 8 * 8 
        #=B * 768 * 8 * 8
        self.up1 = concat_upsample(256 + 512, 32)
        #B * 32 * 16 * 16
        self.up2 = concat_upsample(32 + 256, 32)
        #B * 32 * 32 * 32
        self.up3 = concat_upsample(32 + 128, 32)
        #B * 32 * 64 * 64
        self.up4 = concat_upsample(32 + 64, 32)
        #B * 32 * 128 * 128
        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=(2, 2), stride=2),
            nn.Conv2d(32, 32, (3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            res_block(32, out_channel, stride = 1)
        )
    
    def make_layer(self, in_channel, out_channel, block_num, stride = 2):
        res_layers = []
        res_layers.append(res_block(in_channel, out_channel, stride = stride))
        #for downsample
        for i in range(block_num - 1):
            res_layers.append(res_block(out_channel, out_channel, stride = 1))
        #others
        return nn.Sequential(*res_layers)
    
    def forward(self, x):
        x = self.input_conv(x)
        res1 = self.res_1(x)
        res2 = self.res_2(res1)
        res3 = self.res_3(res2)
        res4 = self.res_4(res3)
        out = self.res_5(res4)
        out = self.up1(res4, out)
        out = self.up2(res3, out)
        out = self.up3(res2, out)
        out = self.up4(res1, out)
        out = self.out_conv(out)
        return out