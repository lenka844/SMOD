import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, attention_module=None):
        super(BasicBlock, self).__init__()
        self.attention_module = attention_module
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        # self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.downsample = downsample
        self.stride = stride
        self.simam = simam_module(planes)

        if attention_module == "simam":
            self.conv2 = nn.Sequential(
                    self.conv2,
                    self.simam
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

    def __init__(self, inplanes, planes, stride=1, heads=4, mhsa=False, resolution=None,
                 downsample=None, groups=1, base_width=64, norm_layer=None, attention_module=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        wid = int(planes * (base_width / 64.)) * groups

        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(inplanes, wid, kernel_size=1, bias=False)
        self.bn1 = norm_layer(wid)
        if not mhsa:
            self.conv2 = nn.Conv2d(wid, wid, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            self.conv2 = nn.ModuleList()
            self.conv2.append(MHSA(wid, width=int(resolution[0]), height=int(resolution[1]), heads=heads))
            if stride == 2:
                self.conv2.append(nn.AdaptiveAvgPool2d(2))
            self.conv2 = nn.Sequential(*self.conv2)
        self.bn2 = nn.BatchNorm2d(wid)
        self.conv3 = nn.Conv2d(wid, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        # self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.simam = simam_module(planes)
        if attention_module == "simam":
            # self.conv2 = self.conv2.append(self.simam)
            self.conv2 = nn.Sequential(
                    self.conv2,
                    self.simam
                    )
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion*planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

        # self.downsample = downsample
        # self.stride = stride

    def forward(self, x):
        # residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # if self.downsample is not None:
        #     residual = self.downsample(x)

        out += self.shortcut(x)
        out = F.relu(out)

        return out


class SimBoT(nn.Module):
    def __init__(self, depth, num_classes=1000, resolution=(64, 64), heads=4, zero_init_residual=False, maxpool=True):
        super(SimBoT, self).__init__()
        blocks = {18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
        layers = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}
        assert layers[depth], 'invalid detph for ResNet (depth should be one of 18, 34, 50, 101, 152, and 200)'

        self.maxpl = maxpool
        self.inplanes = 64
        self.resolution = list(resolution)
        if maxpool:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        if self.conv1.stride[0] == 2:
            self.resolution[0] /= 2
        if self.conv1.stride[1] == 2:
            self.resolution[1] /= 2
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        if maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if self.maxpool.stride == 2:
            self.resolution[0] /= 2
            self.resolution[1] /= 2
        self.layer1 = self._make_layer(blocks[depth], 64, layers[depth][0])
        self.layer2 = self._make_layer(blocks[depth], 128, layers[depth][1], stride=2)
        self.layer3 = self._make_layer(blocks[depth], 256, layers[depth][2], stride=2)
        self.layer4 = self._make_layer(blocks[depth], 512, layers[depth][3], stride=2, heads=heads, mhsa=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * blocks[depth].expansion, num_classes)
        self.fc = nn.Sequential(
            nn.Dropout(0.3), # All architecture deeper than ResNet-200 dropout_rate: 0.2
            nn.Linear(512 * blocks[depth].expansion, num_classes)
        )

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

    def _make_layer(self, block, planes, num_blocks, stride=1, heads=4, mhsa=False):
        # downsample = None
        strides = [stride] + [1]*(num_blocks - 1)
        # if stride != 1 or self.inplanes != planes * block.expansion:
        #     downsample = nn.Sequential(
        #         nn.Conv2d(self.inplanes, planes * block.expansion,
        #                   kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(planes * block.expansion),
        #     )

        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.inplanes, planes, stride, heads, mhsa, self.resolution))
            if stride == 2:
                self.resolution[0] /= 2
                self.resolution[1] /= 2
            self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
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
        return x
