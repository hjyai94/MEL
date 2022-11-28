import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

class ModelBuilder():
    def build_net(self, arch='ml_net', num_input=5, num_classes=5, num_branches=4, padding_list=[0,4,8,12], dilation_list=[2,6,10,14]):
        # parameters in the architecture
        channels = [4, 15, 15, 20, 20, 20, 25, 25, 25, num_classes]
        kernel_size = 3

        # Baselines
        if arch == 'ml_net':
            network = MLNet(channels, kernel_size)
            return network


class BasicBlock(nn.Module):
    def __init__(self, inplanes1, outplanes1, outplanes2, kernel=3, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes1, outplanes1, kernel_size=kernel, dilation=2)
        self.bn1 = nn.BatchNorm3d(outplanes1)
        self.conv2 = nn.Conv3d(outplanes1, outplanes2, kernel_size=kernel, dilation=2)
        self.bn2 = nn.BatchNorm3d(outplanes2)
        self.relu = nn.ReLU(inplace=True)
        if inplanes1 == outplanes2:
            self.downsample = downsample
        else:
            self.downsample = nn.Sequential(nn.Conv3d(inplanes1, outplanes2, kernel_size=1), nn.BatchNorm3d(outplanes2))


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x[:, :, 4:-4, 4:-4, 4:-4]
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

# define the net, called mutual learning net
class MLNet(nn.Module):
    def __init__(self, channels, kernel_size):
        super(MLNet, self).__init__()

        # parameters in the architecture
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
        x = self.conv1(x[:, :, :, :, :])
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

if __name__ == '__main__':
    image = torch.randn(10, 4, 65, 65, 65)
    print(image.shape)
    builder = ModelBuilder()
    model1 = builder.build_net()
    model2 = builder.build_net()
    out1, out2 = model1(image), model2(image)
    print(out1.shape, out2.shape)
    if out1[0, 0, 0, 0, 0] == out2[0, 0, 0, 0, 0]:
        print('equal')
    else:
        print('not equal')



