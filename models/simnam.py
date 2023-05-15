import torch
import torch.nn as nn
import math
from .cbam import *
from .bam import *
from .attention import *


class simam_module(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):

        b, c, h, w = x.size()
        
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, shape, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, attention_module=None, 
                 use_cbam=False, use_nam=False,no_spatial=True):
        super(BasicBlock, self).__init__()
        self.attention_module = attention_module
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        # self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride
        self.no_spatial = no_spatial
        self.simam = simam_module(planes)
        if use_cbam:
            self.cbam = CBAM(planes, 16)
        else:
            self.cbam = None

        if use_nam:
            self.nam = Att(planes,no_spatial=self.no_spatial,shape=shape)
        else:
            self.nam = None

        if attention_module == "simam":
            self.conv2 = nn.Sequential(
                    self.conv2,
                    self.simam
                    )
        elif attention_module == "se":
            self.bn2 = nn.Sequential(
                    self.bn2, 
                    self.se
                    )
        elif attention_module == "ca":
            self.bn2 = nn.Sequential(
                    self.bn2, 
                    self.ca
                    )       

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, shape, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, attention_module=None, 
                 use_cbam=False, use_nam=False, no_spatial=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride
        self.no_spatial = no_spatial

        if use_cbam:
            self.cbam = CBAM(planes * 4, 16)
        else:
            self.cbam = None
        
        if use_nam:
            self.nam = Att(planes * 4, no_spatial=self.no_spatial,shape=shape)
  
        else:
            self.nam = None
        self.simam = simam_module(planes)
        if attention_module == "simam":
            self.conv2 = nn.Sequential(
                    self.conv2,
                    self.simam
                    )
        elif attention_module == "se":
            self.bn3 = nn.Sequential(
                    self.bn3, 
                    self.se
                    )
        elif attention_module == "ca":
            self.bn3 = nn.Sequential(
                    self.bn3, 
                    self.ca
                    )  



    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        
        if not self.cbam is None:
            out = self.cbam(out)

        if not self.nam is None:
            out = self.nam(out)

        out += identity
        out = self.relu(out)

        return out


class SimNAM(nn.Module):
    def __init__(self, depth, num_classes=1000, zero_init_residual=False, maxpool=True, att_type=None):
        super().__init__()
        blocks = {18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
        layers = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}
        assert layers[depth], 'invalid detph for ResNet (depth should be one of 18, 34, 50, 101, 152, and 200)'

        self.maxpl = maxpool
        self.inplanes = 64
        if maxpool:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(7)
        shape = 56
        if maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if att_type == 'BAM':
            self.bam1 = BAM(64*block.expansion)
            self.bam2 = BAM(128*block.expansion)
            self.bam3 = BAM(256*block.expansion)
        else:
            self.bam1, self.bam2, self.bam3 = None, None, None
        self.layer1 = self._make_layer(blocks[depth], 64, shape, layers[depth][0], att_type=att_type, no_spatial=False)
        self.layer2 = self._make_layer(blocks[depth], 128, shape//2, layers[depth][1], stride=2, att_type=att_type, no_spatial=False)
        self.layer3 = self._make_layer(blocks[depth], 256, shape//4, layers[depth][2], stride=2, att_type=att_type, no_spatial=False)
        self.layer4 = self._make_layer(blocks[depth], 512, shape//8, layers[depth][3], stride=2, att_type=att_type, no_spatial=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * blocks[depth].expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, shape, blocks, stride=1, att_type=None, no_spatial=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, shape, stride, downsample, use_cbam=att_type == 'CBAM', use_nam=att_type == 'NAM',
                            no_spatial=no_spatial))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, shape,use_cbam=att_type == 'CBAM', use_nam=att_type == 'NAM',
                                no_spatial=no_spatial))

        return nn.Sequential(*layers)

    def forward(self, x, return_feat=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpl:
            x = self.maxpool(x)

        x = self.layer1(x)
        if not self.bam1 is None:
            x = self.bam1(x)
        x = self.layer2(x)
        if not self.bam2 is None:
            x = self.bam2(x)
        x = self.layer3(x)
        if not self.bam3 is None:
            x = self.bam3(x)
        x = self.layer4(x)
        if return_feat:
            return x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x