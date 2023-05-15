from multiprocessing import reduction
from turtle import forward
from sklearn.metrics import recall_score
import torch
import torch.nn as nn
from numpy import outer
import math

class GAM_Attention(nn.Module):
    def __init__(self, in_channels, rate=4):
        super(GAM_Attention, self).__init__()
        self.channel_attention = nn.Sequential(  
            nn.Linear(in_channels, int(in_channels / rate)),  
            nn.ReLU(inplace=True),  
            nn.Linear(int(in_channels / rate), in_channels)  
        )

        self.spatial_attention = nn.Sequential(  
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),  
            nn.BatchNorm2d(int(in_channels / rate)),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(int(in_channels / rate), in_channels, kernel_size=7, padding=3),  
            nn.BatchNorm2d(in_channels)  
        )
    
    def forward(self, x):
        b, c, h, w = x.size()
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att
        x_spatial_att = self.spatial_attention(x).sigmoid()

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, bias=False, groups=groups, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class GAMBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(GAMBasicBlock, self).__init__()
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


        # self.ca = ChannelAttention(planes)
        # self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # out = self.ca(out) * out
        # out = self.sa(out) * out


        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class GAMBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(GAMBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        # self.ca = ChannelAttention(planes * self.expansion)
        # self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

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

        # out = self.ca(out) * out
        # out = self.sa(out) * out

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class GAMResNet(nn.Module):
    def __init__(self, depth, num_classes=1000, zero_init_residual=False, maxpool=True, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(GAMResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        blocks = {18: GAMBasicBlock, 34: GAMBasicBlock, 50: GAMBottleneck, 101: GAMBottleneck, 152: GAMBottleneck, 200: GAMBottleneck}
        layers = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}
        assert layers[depth], 'invalid detph for ResNet (depth should be one of 18, 34, 50, 101, 152, and 200)'
        self.maxpl = maxpool
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None'
                             'or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        if maxpool:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.gam = GAM_Attention(self.inplanes)
        # self.ca = ChannelAttention(self.inplanes)
        # self.sa = SpatialAttention()
        if maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(blocks[depth], 64, layers[depth][0])
        self.layer2 = self._make_layer(blocks[depth], 128, layers[depth][1], stride=2)
        self.layer3 = self._make_layer(blocks[depth], 256, layers[depth][2], stride=2)
        self.layer4 = self._make_layer(blocks[depth], 512, layers[depth][3], stride=2)
        # self.ca1 = ChannelAttention(self.inplanes)
        # self.sa1 = SpatialAttention()
        self.gam1 = GAM_Attention(self.inplanes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * blocks[depth].expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, GAMBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, GAMBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride= 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes* block.expansion)
            )
            # downsample = nn.Sequential(
            #     nn.Conv2d(self.inplanes, planes * block.expansion,
            #               kernel_size=1, stride=stride, bias=False),
            #     nn.BatchNorm2d(planes * block.expansion),
            # )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, return_feat=False):
        # print('------using SEBlocks-------')
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # x = self.ca(x) * x
        # x = self.sa(x) * x
        x = self.gam(x) * x

        if self.maxpl:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.ca1(x) * x
        # x= self.sa1(x) * x
        x = self.gam1(x) * x
        if return_feat:
            return x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x