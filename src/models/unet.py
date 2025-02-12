# Implement your UNet model here
import torch.nn as nn
import torch
import torch.nn.init as init
import math

class double_conv_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(double_conv_block, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (3, 3),padding=1),
            #add padding for same dim for output
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            #nn.ReflectionPad2d(1),
            nn.Conv2d(out_channel, out_channel, (3, 3), padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
            )
        
        for layer in self.double_conv:
            if isinstance(layer, nn.Conv2d):
                init.normal_(layer.weight, mean=0.0, std=math.sqrt(2/(9 * layer.in_channels)))
                nn.init.constant_(layer.bias, 0)
        
    def forward(self, x):
        torch.cuda.synchronize()
        print("DoubleConv input shape:", x.shape)
        x = self.double_conv(x)
        torch.cuda.synchronize()
        print("DoubleConv output shape:", x.shape)
        return x

class down_conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(down_conv, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.double_conv = double_conv_block(in_channel, out_channel)
        
    def forward(self, x):
        x = self.maxpool(x)
        x = self.double_conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(up_conv, self).__init__()
        self.up_convolution = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=(2, 2), stride=2)
        init.normal_(self.up_convolution.weight, mean=1, std= math.sqrt(2/(9 * in_channel)))
        init.constant_(self.up_convolution.bias, 0)
        
        self.double_conv = double_conv_block(in_channel, out_channel)
        
    def forward(self, res_x, x):
        x = self.up_convolution(x)
        x = torch.cat([res_x, x], dim=1)
        x = self.double_conv(x)
        return x

class Unet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Unet, self).__init__()
        self.in_conv = double_conv_block(in_channel, 64)
        self.down1 = down_conv(64,128)
        self.down2 = down_conv(128, 256)
        self.down3 = down_conv(256, 512)
        self.down4 = down_conv(512, 1024)
        self.up1 = up_conv(1024, 512)
        self.up2 = up_conv(512, 256)
        self.up3 = up_conv(256, 128)
        self.up4 = up_conv(128, 64)
        self.out_conv = nn.Conv2d(64, out_channel, (1, 1))
    def forward(self, x):
        res1 = self.in_conv(x)
        res2 = self.down1(res1)
        res3 = self.down2(res2)
        res4 = self.down3(res3)
        x = self.down4(res4)
        print(x.shape)
        print(res4.shape)
        x = self.up1(res4, x)
        print(x.shape)
        print(res3.shape)
        x = self.up2(res3, x)
        print(x.shape)
        x = self.up3(res2, x)
        x = self.up4(res1, x)
        logits = self.out_conv(x)
        return logits

        
    
        
        
        