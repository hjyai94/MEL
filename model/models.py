import torch.nn as nn 
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from ptflops import get_model_complexity_info
from thop import profile

class ModelBuilder():
    def build_net(self, arch='Basic', num_input=5, num_classes=5, num_branches=4, padding_list=[0,4,8,12], dilation_list=[2,6,10,14]):
        # parameters in the architecture
        channels = [4, 30, 30, 40, 40, 40, 40, 50, 50, num_classes]
        kernel_size = 3

        # Baselines
        if arch == 'DM':
            network = Basic(channels, kernel_size)
            return network
        elif arch == 'Unet':
            network = Unet()
            return network
        else:
            raise Exception('Architecture undefined')



class BasicBlock(nn.Module):
    def __init__(self, inplanes1, outplanes1, outplanes2, kernel=3, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 =nn.Conv3d(inplanes1, outplanes1, kernel_size=kernel,dilation=2)
        self.bn1 = nn.BatchNorm3d(outplanes1)
        self.conv2 =nn.Conv3d(outplanes1, outplanes2, kernel_size=kernel, dilation=2)
        self.bn2 = nn.BatchNorm3d(outplanes2)
        self.relu = nn.ReLU(inplace=True)
        if inplanes1==outplanes2:            
            self.downsample = downsample
        else:
            self.downsample = nn.Sequential(nn.Conv3d(inplanes1, outplanes2, kernel_size=1), nn.BatchNorm3d(outplanes2))
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self,x):
        residual = x[:,:, 4:-4, 4:-4, 4:-4]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
    
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        x += residual
        x = self.relu(x)
        return x
    

class Basic(nn.Module):
    def __init__(self, channels, kernel_size):
        super(Basic, self).__init__()
        
        # parameters in the architecture
        self.channels = channels     
        self.kernel_size = kernel_size
        
        # network architecture
        self.conv1 = nn.Conv3d(self.channels[0], self.channels[1], kernel_size=self.kernel_size)
        self.bn1 = nn.BatchNorm3d(self.channels[1])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(self.channels[1], self.channels[2], kernel_size=self.kernel_size)
        self.bn2 = nn.BatchNorm3d(self.channels[2])
        
        self.layer3 = BasicBlock(self.channels[2], self.channels[3], self.channels[4])
        self.layer4 = BasicBlock(self.channels[4], self.channels[5], self.channels[6])
        self.layer5 = BasicBlock(self.channels[6], self.channels[7], self.channels[8])
           
        self.fc = nn.Conv3d(self.channels[8], self.channels[9], kernel_size=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.layer3(x)
        x = self.layer4(x)        
        x = self.layer5(x)
              
        x = self.fc(x)
        return x



# U-Net
def normalization(planes, norm='gn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m


class ConvD(nn.Module):
    def __init__(self, inplanes, planes, dropout=0.0, norm='gn', first=False):
        super(ConvD, self).__init__()

        self.first = first
        self.maxpool = nn.MaxPool3d(2, 2, ceil_mode=True)

        self.dropout = dropout
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = normalization(planes, norm)

        self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn3   = normalization(planes, norm)

    def forward(self, x):
        if not self.first:
            x = self.maxpool(x)
        x = self.bn1(self.conv1(x))
        y = self.relu(self.bn2(self.conv2(x)))
        if self.dropout > 0:
            y = F.dropout3d(y, self.dropout)
        y = self.bn3(self.conv3(x))
        return self.relu(x + y)


class ConvU(nn.Module):
    def __init__(self, planes, norm='gn', first=False):
        super(ConvU, self).__init__()

        self.first = first

        if not self.first:
            self.conv1 = nn.Conv3d(2*planes, planes, 3, 1, 1, bias=False)
            self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv3d(planes, planes//2, 1, 1, 0, bias=False)
        self.bn2   = normalization(planes//2, norm)

        self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        self.bn3   = normalization(planes, norm)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, prev):
        # final output is the localization layer
        if not self.first:
            x = self.relu(self.bn1(self.conv1(x)))

        y = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        y = self.relu(self.bn2(self.conv2(y)))

        diffH = prev.size()[2] - y.size()[2]
        diffW = prev.size()[3] - y.size()[3]
        diffD = prev.size()[4] - y.size()[4]

        y = F.pad(y, (diffW//2, diffW-diffW//2, diffH//2, diffH-diffH//2, diffD//2, diffD-diffD//2))
        y = torch.cat([prev, y], 1)
        y = self.relu(self.bn3(self.conv3(y)))

        return y


class Unet(nn.Module):
    def __init__(self, c=4, n=16, dropout=0.5, norm='gn', num_classes=5):
        super(Unet, self).__init__()
        # self.upsample = F.interpolate(scale_factor=2,
        #         mode='trilinear', align_corners=False)

        self.convd1 = ConvD(c,     n, dropout, norm, first=True)
        self.convd2 = ConvD(n,   2*n, dropout, norm)
        self.convd3 = ConvD(2*n, 4*n, dropout, norm)
        self.convd4 = ConvD(4*n, 8*n, dropout, norm)
        self.convd5 = ConvD(8*n,16*n, dropout, norm)

        self.convu4 = ConvU(16*n, norm, True)
        self.convu3 = ConvU(8*n, norm)
        self.convu2 = ConvU(4*n, norm)
        self.convu1 = ConvU(2*n, norm)

        self.seg3 = nn.Conv3d(8*n, num_classes, 1)
        self.seg2 = nn.Conv3d(4*n, num_classes, 1)
        self.seg1 = nn.Conv3d(2*n, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.convd1(x)
        x2 = self.convd2(x1)
        x3 = self.convd3(x2)
        x4 = self.convd4(x3)
        x5 = self.convd5(x4)

        y4 = self.convu4(x5, x4)
        y3 = self.convu3(y4, x3)
        y2 = self.convu2(y3, x2)
        y1 = self.convu1(y2, x1)

        y3 = self.seg3(y3)
        y2 = self.seg2(y2) + F.interpolate(y3, scale_factor=2, mode='trilinear', align_corners=False)
        y1 = self.seg1(y1) + F.interpolate(y2, scale_factor=2, mode='trilinear', align_corners=False)

        return y1

if __name__ == '__main__':
    image = torch.randn(1, 4, 64, 64, 64)
    print(image.shape)
    builder = ModelBuilder()
    model1 = builder.build_net('Unet')
    model2 = builder.build_net('Basic')
    # out1, out2 = model1(image), model2(image)
    # print(out1.shape, out2.shape)
    # if out1[0, 0, 0, 0, 0] == out2[0, 0, 0, 0, 0]:
    #     print('equal')
    # else:
    #     print('not equal')

    flops1, params1 = get_model_complexity_info(model1, (4, 64, 64, 64), as_strings=True, print_per_layer_stat=False)
    print('{:<30}  {:<8}'.format('Model1 Computational complexity: ', flops1))
    print('{:<30}  {:<8}'.format('Model1 Number of parameters: ', params1))

    flops2, params2 = get_model_complexity_info(model2, (4, 64, 64, 64), as_strings=True, print_per_layer_stat=False)
    print('{:<30}  {:<8}'.format('Model2 Computational complexity: ', flops2))
    print('{:<30}  {:<8}'.format('Model2 Number of parameters: ', params2))

    # flops, params = profile(model2, inputs=(image,))
    # print('FLOPs = ' + str(flops/1000**3) + 'G')
    # print('Params = ' + str(params/1000**2) + 'M')
