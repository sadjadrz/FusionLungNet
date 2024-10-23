import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            m.initialize()


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out      = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out      = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out      = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(out+residual, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)
        self.initialize()

    def make_layer(self, planes, blocks, stride, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes*4:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))

        layers = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torchvision.models.resnet50(True).state_dict(), strict=False)


class CAA(nn.Module):
    def __init__(self, in_channel_left, in_channel_right):
        super(CAA, self).__init__()
        self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=1, stride=1, padding=0)
        self.bn0   = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(in_channel_right, 256, kernel_size=1, stride=1, padding=0)

        self.conv13 = nn.Conv2d(256, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv31 = nn.Conv2d(256, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))

    def forward(self, left, down):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True)
        left = F.relu(self.bn1(self.conv2(left)), inplace=True)
        down = F.relu(self.conv1(down), inplace=True)
        down = F.relu(self.conv31(down), inplace=True)
        down = F.relu(self.conv13(down), inplace=True)
        down = torch.sigmoid(down.mean(dim=(2, 3), keepdim=True))
        return left * down

    def initialize(self):
        weight_init(self)


class MFF(nn.Module):
    def __init__(self, in_channel_left, in_channel_down, in_channel_right):
        super(MFF, self).__init__()
        self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(in_channel_down, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(in_channel_right, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256 * 3, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, left, down, right):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True)  # 256 channels
        down = F.relu(self.bn1(self.conv1(down)), inplace=True)  # 256 channels
        right = F.relu(self.bn2(self.conv2(right)), inplace=True)  # 256 channels

        down = F.interpolate(down, size=left.size()[2:], mode='bilinear')
        right = F.interpolate(right, size=left.size()[2:], mode='bilinear')

        x = left * down # l*h
        y = left * right # l*c
        z = right * down # h*c
        out = torch.cat([x, y, z], dim=1)
        return F.relu(self.bn3(self.conv3(out)), inplace=True)

    def initialize(self):
        weight_init(self)


class SR(nn.Module):
    def __init__(self, in_channel):
        super(SR, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True) #256
        out2 = self.conv2(out1)
        w, b = out2[:, :256, :, :], out2[:, 256:, :, :]
        return F.relu(w * out1 + b, inplace=True)

    def initialize(self):
        weight_init(self)

class RefUnet(nn.Module):
    def __init__(self, in_ch, inc_ch):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch, inc_ch, 3, padding=1)

        self.conv1 = nn.Conv2d(inc_ch, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        #self.relu1 = nn.ReLU(inplace=True)

        #self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        #self.relu2 = nn.ReLU(inplace=True)

        #self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        #self.relu3 = nn.ReLU(inplace=True)

        #self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        #self.relu4 = nn.ReLU(inplace=True)

        #self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        #self.relu5 = nn.ReLU(inplace=True)

        #####

        self.conv_d4 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        #self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        #self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        #self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        #self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64, 1, 3, padding=1)

        #self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        hx = x
        hx = self.conv0(hx)
        hx1 = F.relu(self.bn1(self.conv1(hx)))
        hx = F.max_pool2d(hx1, kernel_size=2, stride=2, ceil_mode=True)

        hx2 = F.relu(self.bn2(self.conv2(hx)))
        hx =  F.max_pool2d(hx2, kernel_size=2, stride=2, ceil_mode=True)

        hx3 = F.relu(self.bn3(self.conv3(hx)))
        hx = F.max_pool2d(hx3, kernel_size=2, stride=2, ceil_mode=True)

        hx4 = F.relu(self.bn4(self.conv4(hx)))
        hx = F.max_pool2d(hx4, kernel_size=2, stride=2, ceil_mode=True)

        hx5 = F.relu(self.bn5(self.conv5(hx)))
        #hx = self.upscore2(hx5)
        hx = F.interpolate(hx5, scale_factor=2, mode='bilinear')
        

        d4 = F.relu(self.bn_d4(self.conv_d4(torch.cat((hx, hx4), 1))))
        hx = F.interpolate(d4, scale_factor=2, mode='bilinear')
        #hx = self.upscore2(d4)

        d3 = F.relu(self.bn_d3(self.conv_d3(torch.cat((hx, hx3), 1))))
        #hx = self.upscore2(d3)
        hx = F.interpolate(d3, scale_factor=2, mode='bilinear')

        d2 = F.relu(self.bn_d2(self.conv_d2(torch.cat((hx, hx2), 1))))
        hx = F.interpolate(d2, scale_factor=2, mode='bilinear')
        #hx = self.upscore2(d2)

        d1 = F.relu(self.bn_d1(self.conv_d1(torch.cat((hx, hx1), 1))))

        residual = self.conv_d0(d1)
        return x + residual
    def initialize(self):
        weight_init(self)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
class FusionLungNet(nn.Module):
    def __init__(self, cfg):
        super(FusionLungNet, self).__init__()
        self.cfg     = cfg
        self.bkbone  = ResNet()
        self.ca55 = CAA(2048, 2048)

        self.fam45 = MFF(1024,  256, 256)
        self.fam34 = MFF(512,  256, 256)
        self.fam23 = MFF(256,  256, 256)

        self.srm5 = SR(256)
        self.srm4 = SR(256)
        self.srm3 = SR(256)
        self.srm2 = SR(256)

        self.linear5 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)

                ## -------------Refine Module-------------
        self.refunet = RefUnet(1, 64)

        self.initialize()

    def forward(self, x):
        out1, out2, out3, out4, out5_ = self.bkbone(x)
        out5 = self.ca55(out5_, out5_)
        out5 = self.srm5(out5)
        out4 = self.srm4(self.fam45(out4, out5, out5))
        out3 = self.srm3(self.fam34(out3, out4, out5))
        out2 = self.srm2(self.fam23(out2, out3, out5)) 

        out5 = F.interpolate(self.linear5(out5), size=x.size()[2:], mode='bilinear')
        out4 = F.interpolate(self.linear4(out4), size=x.size()[2:], mode='bilinear')
        out3 = F.interpolate(self.linear3(out3), size=x.size()[2:], mode='bilinear')
        out2 = F.interpolate(self.linear2(out2), size=x.size()[2:], mode='bilinear')
        dout = self.refunet(out2)  

        return F.sigmoid(dout), F.sigmoid(out2), F.sigmoid(out3), F.sigmoid(out4), F.sigmoid(out5)
        #return out2, out3, out4, out5

    def initialize(self):
        if self.cfg:
            try:
                print("load params")
                self.load_state_dict(torch.load(self.cfg.snapshot), strict=True)
            except:
                print("Warning: please check the snapshot file:", self.cfg.snapshot)
                pass
        else:
            weight_init(self)



