import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.hub import load_state_dict_from_url
import torchvision
from functools import partial
from collections import OrderedDict
import math
from models.binary3dresnet import attentionresnet34


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1   = nn.Conv3d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv3d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



class BasicBlock(nn.Module):
    expansion = 1
    # planes refer to the number of feature maps
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.downsample = downsample
        self.conv1 = nn.Conv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        # conv1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # conv2
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        # downsample
        if self.downsample is not None:
            residual = self.downsample(x)

        # print(out.shape, residual.shape)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    # planes refer to the number of feature maps
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.downsample = downsample
        self.conv1 = nn.Conv3d(
            inplanes, planes, kernel_size=1, bias=False) # kernal_size=1 don't need padding
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()
    def forward(self, x):
        residual = x
        # conv1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # conv2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # conv3
        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        # downsample
        if self.downsample is not None:
            residual = self.downsample(x)

        # print(out.shape, residual.shape)
        out += residual
        out = self.relu(out)

        return out


def downsample_basic_block(x, planes, stride):
    # decrease data resolution if stride not equals to 1
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    # shape: (batch_size, channel, t, h, w)
    # try to match the channel size
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class ResNet(nn.Module):
    def __init__(self, block, layers, shortcut_type, sample_size, sample_duration, num_classes=10):
        super(ResNet, self).__init__()
        # initialize inplanes to 64, it'll be changed later
        self.inplanes = 64
        self.sigmoid = nn.Sigmoid()
        self.binary_resnet=attentionresnet34(pretrained=False,sample_size=sample_size, sample_duration=sample_duration, attention=False)
        self.sa = SpatialAttention()
        self.conv1 = nn.Conv3d(
            3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        # layers refers to the number of blocks in each layer
        self.layer1 = self._make_layer(
            block, 64, layers[0], shortcut_type, stride=1)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2)
        # calclatue kernal size for average pooling
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        # attention block
        # init the weights
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride):
        downsample = None
        # when the in-channel and the out-channel dismatch, downsample!!!
        if stride != 1 or self.inplanes != planes * block.expansion:
            # stride once for downsample and block.
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        # only the first block needs downsample.
        layers.append(block(self.inplanes, planes, stride, downsample))
        # change inplanes for the next layer
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, y):
        attention=self.binary_resnet(y)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)
        att_l1 = ChannelAttention(64)(l1) * l1
        attention1=self.sa(l1)
        l1 = self.sigmoid(self.sa(att_l1) + attention[0])*att_l1

        l2 = self.layer2(l1)
        att_l2 = ChannelAttention(128)(l2) * l2
        attention2=self.sa(l2)
        l2 = self.sigmoid(self.sa(att_l2) + attention[1])*att_l2

        l3 = self.layer3(l2)
        att_l3 = ChannelAttention(256)(l3) * l3
        attention3=self.sa(l3)
        l3 = self.sigmoid(self.sa(att_l3) + attention[2])*att_l3

        l4 = self.layer4(l3)
        att_l4 = ChannelAttention(512)(l4) * l4
        attention4=self.sa(l4)
        l4 = self.sigmoid(self.sa(att_l4) + attention[3])*att_l4

        g = self.avgpool(l4)
        # attention

        c1, c2, c3, c4 = None, None, None, None
            # x.size(0) ------ batch_size
        g = g.view(g.size(0), -1)
        x = self.fc(g)
        #return g
        # return [x, c1, c2, c3, c4]
        return x
    def load_my_state_dict(self, state_dict):
        my_state_dict = self.state_dict()
        for name, param in state_dict.items():
            if name == 'fc.weight' or name == 'fc.bias':
                continue
            my_state_dict[name].copy_(param.data)


model_urls = {
    'resnet18': 'https://www.jianguoyun.com/c/dl-file/resnet-18-kinetics.pth?dt=q67aev&kv=YXF6QHpqdS5lZHUuY24&sd=a54cr&ud=B8Sbfz0nRv1pG8YNAbo0KiCnzvJHDsLYQsWjtT4b1j8&vr=1',
    'resnet34': 'https://www.jianguoyun.com/c/dl-file/resnet-34-kinetics.pth?dt=q67acv&kv=YXF6QHpqdS5lZHUuY24&sd=a54cr&ud=BftTcvolMjyywptfxelwwjXJksCaU0ektvfMwCbMD1I&vr=1',
    'resnet50': 'https://www.jianguoyun.com/c/dl-file/resnet-50-kinetics.pth?dt=q67atr&kv=YXF6QHpqdS5lZHUuY24&sd=a54cr&ud=uKpTbIK63qX3bHs2weOGqYYc2gtssQi-o7UqpoTaG6Q&vr=1',
    'resnet101': '',
    'resnet152': '',
    'resnet200': '',
}


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], shortcut_type='A', **kwargs)
    if pretrained:
        checkpoint = torch.load('C:\\Users\\han006\\Desktop\\plan_A_code_v3\\pretrained_weight\\resnet-18-kinetics.pth')
        state_dict = checkpoint['state_dict']

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module.'
            new_state_dict[name]=v
        model.load_my_state_dict(new_state_dict)

    return model


def resnet34(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], shortcut_type='A', **kwargs)
    if pretrained:
        checkpoint = load_state_dict_from_url(model_urls['resnet34'],
            progress=progress)
        state_dict = checkpoint['state_dict']

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module.'
            new_state_dict[name]=v
        model.load_my_state_dict(new_state_dict)

    return model


def resnet50(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], shortcut_type='B', **kwargs)
    if pretrained:
        checkpoint = load_state_dict_from_url(model_urls['resnet50'],
            progress=progress)
        state_dict = checkpoint['state_dict']

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module.'
            new_state_dict[name]=v
        model.load_my_state_dict(new_state_dict)

    return model


def resnet101(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], shortcut_type='B', **kwargs)
    if pretrained:
        checkpoint = load_state_dict_from_url(model_urls['resnet101'],
            progress=progress)
        state_dict = checkpoint['state_dict']

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module.'
            new_state_dict[name]=v
        model.load_my_state_dict(new_state_dict)

    return model


def resnet152(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], shortcut_type='B', **kwargs)
    if pretrained:
        checkpoint = load_state_dict_from_url(model_urls['resnet152'],
            progress=progress)
        state_dict = checkpoint['state_dict']

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module.'
            new_state_dict[name]=v
        model.load_my_state_dict(new_state_dict)

    return model


def resnet200(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], shortcut_type='B', **kwargs)
    if pretrained:
        checkpoint = load_state_dict_from_url(model_urls['resnet200'],
            progress=progress)
        state_dict = checkpoint['state_dict']

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module.'
            new_state_dict[name]=v
        model.load_my_state_dict(new_state_dict)

    return model

if __name__ == '__main__':
    import sys
    sys.path.append("..")
    import torchvision.transforms as transforms
    import torch
    #print(torch.load('C:\\Users\\han006\\Desktop\\plan_A_code_v3\\pretrained_weight\\resnet-18-kinetics.pth'))
    #from dataset import CSL_Isolated
    sample_size = 128
    sample_duration = 16
    num_classes = 500
    transform = transforms.Compose([transforms.Resize([sample_size, sample_size]), transforms.ToTensor()])
    # dataset = CSL_Isolated(data_path="/home/haodong/Data/CSL_Isolated/color_video_125000",
    #     label_path="/home/haodong/Data/CSL_Isolated/dictionary.txt", frames=sample_duration,
    #     num_classes=num_classes, transform=transform)
    # cnn3d = CNN3D(sample_size=sample_size, sample_duration=sample_duration, num_classes=num_classes)
    data=torch.rand(1,3,16,128,128)
    data2=torch.rand(1,3,16,128,128)
    cnn3d = resnet18(pretrained=False,sample_size=sample_size, sample_duration=sample_duration, num_classes=10)
    print(cnn3d(data,data2))