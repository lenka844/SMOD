import torch
import torch.nn as nn
from functools import reduce

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class SKConv(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,M=2,r=16,L=32):
        '''
        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param stride:  步长，默认为1
        :param M:  分支数
        :param r: 特征Z的长度，计算其维度d 时所需的比率（论文中 特征S->Z 是降维，故需要规定 降维的下界）
        :param L:  论文中规定特征Z的下界，默认为32
        采用分组卷积： groups = 32,所以输入channel的数值必须是group的整数倍
        '''
        super(SKConv,self).__init__()
        d=max(in_channels//r,L)   # 计算从向量C降维到 向量Z 的长度d
        self.M=M
        self.out_channels=out_channels
        self.conv=nn.ModuleList()  # 根据分支数量 添加 不同核的卷积操作
        for i in range(M):
            # 为提高效率，原论文中 扩张卷积5x5为 （3X3，dilation=2）来代替。 且论文中建议组卷积G=32
            self.conv.append(nn.Sequential(nn.Conv2d(in_channels,out_channels,3,stride,padding=1+i,dilation=1+i,groups=32,bias=False),
                                           nn.BatchNorm2d(out_channels),
                                           nn.ReLU(inplace=True)))
        self.global_pool=nn.AdaptiveAvgPool2d(output_size = 1) # 自适应pool到指定维度    这里指定为1，实现 GAP
        self.fc1=nn.Sequential(nn.Conv2d(out_channels,d,1,bias=False),
                               nn.BatchNorm2d(d),
                               nn.ReLU(inplace=True))   # 降维
        self.fc2=nn.Conv2d(d,out_channels*M,1,1,bias=False)  # 升维
        self.softmax=nn.Softmax(dim=1) # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1
    def forward(self, input):
        batch_size=input.size(0)
        output=[]
        #the part of split
        for i,conv in enumerate(self.conv):
            output.append(conv(input))    #[batch_size,out_channels,H,W]
        #the part of fusion
        U=reduce(lambda x,y:x+y,output) # 逐元素相加生成 混合特征U  [batch_size,channel,H,W]         
        s=self.global_pool(U)     # [batch_size,channel,1,1]
        z=self.fc1(s)  # S->Z降维   # [batch_size,d,1,1]
        a_b=self.fc2(z) # Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b   [batch_size,out_channels*M,1,1]
        a_b=a_b.reshape(batch_size,self.M,self.out_channels,-1) #调整形状，变为 两个全连接层的值[batch_size,M,out_channels,1]  
        a_b=self.softmax(a_b) # 使得两个全连接层对应位置进行softmax [batch_size,M,out_channels,1]  
        #the part of selection
        a_b=list(a_b.chunk(self.M,dim=1))#split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块 [[batch_size,1,out_channels,1],[batch_size,1,out_channels,1]
        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b)) # 将所有分块  调整形状，即扩展两维  [[batch_size,out_channels,1,1],[batch_size,out_channels,1,1]
        V=list(map(lambda x,y:x*y,output,a_b)) # 权重与对应  不同卷积核输出的U 逐元素相乘[batch_size,out_channels,H,W] * [batch_size,out_channels,1,1] = [batch_size,out_channels,H,W]
        V=reduce(lambda x,y:x+y,V) # 两个加权后的特征 逐元素相加  [batch_size,out_channels,H,W] + [batch_size,out_channels,H,W] = [batch_size,out_channels,H,W]
        return V    # [batch_size,out_channels,H,W]



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SKBlock(nn.Module):
    '''
    基于Res Block构造的SK Block
    ResNeXt有  1x1Conv（通道数：x） +  SKConv（通道数：x）  + 1x1Conv（通道数：2x） 构成
    '''
    expansion=2 #指 每个block中 通道数增大指定倍数
    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(SKBlock,self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(inplanes,planes,1,1,0,bias=False),
                                 nn.BatchNorm2d(planes),
                                 nn.ReLU(inplace=True))
        self.conv2=SKConv(planes,planes,stride)
        # 与 ResNet block最大的区别就在于中间的这个3*3的卷积 使用 SkConv进行取代
        self.conv3=nn.Sequential(nn.Conv2d(planes,planes*self.expansion,1,1,0,bias=False),
                                 nn.BatchNorm2d(planes*self.expansion))
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample
    def forward(self, input):
        shortcut=input
        output=self.conv1(input)
        output=self.conv2(output)
        output=self.conv3(output)
        if self.downsample is not None:
            shortcut=self.downsample(input)
        output+=shortcut
        return self.relu(output)


class SKNet(nn.Module):
    def __init__(self, depth, num_classes=1000, zero_init_residual=False, maxpool=True):
        super(SKNet, self).__init__()
        blocks = {18: BasicBlock, 34: BasicBlock, 50: SKBlock, 101: SKBlock, 152: SKBlock, 200: SKBlock}
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
        if maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(blocks[depth], 128, layers[depth][0])
        self.layer2 = self._make_layer(blocks[depth], 256, layers[depth][1], stride=2)
        self.layer3 = self._make_layer(blocks[depth], 512, layers[depth][2], stride=2)
        self.layer4 = self._make_layer(blocks[depth], 1024, layers[depth][3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024 * blocks[depth].expansion, num_classes)
        self.softmax=nn.Softmax(-1)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, return_feat=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpl:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if return_feat:
            return x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.softmax(x)
        return x