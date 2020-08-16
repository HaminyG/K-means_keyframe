# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# from torch.hub import load_state_dict_from_url
# import torchvision
# from functools import partial
# from collections import OrderedDict
# import math
# 
# 
# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool3d(1)
#         self.max_pool = nn.AdaptiveMaxPool3d(1)
# 
#         self.fc1   = nn.Conv3d(in_planes, in_planes // 16, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2   = nn.Conv3d(in_planes // 16, in_planes, 1, bias=False)
# 
#         self.sigmoid = nn.Sigmoid()
# 
#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
# 
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
# 
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
# 
#         self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
# 
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)
# 
# 
# 
# class BasicBlock(nn.Module):
#     expansion = 1
#     # planes refer to the number of feature maps
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.stride = stride
#         self.downsample = downsample
#         self.conv1 = nn.Conv3d(
#             inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm3d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv3d(
#             planes, planes, kernel_size=3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm3d(planes)
# 
#         self.ca = ChannelAttention(planes)
#         self.sa = SpatialAttention()
# 
#     def forward(self, x):
#         residual = x
#         # conv1
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         # conv2
#         out = self.conv2(out)
#         out = self.bn2(out)
# 
#         out = self.ca(out) * out
#         out = self.sa(out) * out
# 
#         # downsample
#         if self.downsample is not None:
#             residual = self.downsample(x)
# 
#         # print(out.shape, residual.shape)
#         out += residual
#         out = self.relu(out)
# 
#         return out
# 
# 
# class Bottleneck(nn.Module):
#     expansion = 4
#     # planes refer to the number of feature maps
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.stride = stride
#         self.downsample = downsample
#         self.conv1 = nn.Conv3d(
#             inplanes, planes, kernel_size=1, bias=False) # kernal_size=1 don't need padding
#         self.bn1 = nn.BatchNorm3d(planes)
#         self.conv2 = nn.Conv3d(
#             planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm3d(planes)
#         self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm3d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.ca = ChannelAttention(planes * 4)
#         self.sa = SpatialAttention()
#     def forward(self, x):
#         residual = x
#         # conv1
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         # conv2
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#         # conv3
#         out = self.conv3(out)
#         out = self.bn3(out)
# 
#         out = self.ca(out) * out
#         out = self.sa(out) * out
# 
#         # downsample
#         if self.downsample is not None:
#             residual = self.downsample(x)
# 
#         # print(out.shape, residual.shape)
#         out += residual
#         out = self.relu(out)
# 
#         return out
# 
# 
# def downsample_basic_block(x, planes, stride):
#     # decrease data resolution if stride not equals to 1
#     out = F.avg_pool3d(x, kernel_size=1, stride=stride)
#     # shape: (batch_size, channel, t, h, w)
#     # try to match the channel size
#     zero_pads = torch.Tensor(
#         out.size(0), planes - out.size(1), out.size(2), out.size(3),
#         out.size(4)).zero_()
#     if isinstance(out.data, torch.cuda.FloatTensor):
#         zero_pads = zero_pads.cuda()
# 
#     out = Variable(torch.cat([out.data, zero_pads], dim=1))
# 
#     return out
# 
# 
# class ResNet(nn.Module):
#     def __init__(self, block, layers, shortcut_type, sample_size, sample_duration, num_classes=500):
#         super(ResNet, self).__init__()
#         # initialize inplanes to 64, it'll be changed later
#         self.inplanes = 64
#         self.conv1 = nn.Conv3d(
#             3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
#         self.bn1 = nn.BatchNorm3d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
#         # layers refers to the number of blocks in each layer
#         self.layer1 = self._make_layer(
#             block, 64, layers[0], shortcut_type, stride=1)
#         self.layer2 = self._make_layer(
#             block, 128, layers[1], shortcut_type, stride=2)
#         self.layer3 = self._make_layer(
#             block, 256, layers[2], shortcut_type, stride=2)
#         self.layer4 = self._make_layer(
#             block, 512, layers[3], shortcut_type, stride=2)
#         # calclatue kernal size for average pooling
#         last_duration = int(math.ceil(sample_duration / 16))
#         last_size = int(math.ceil(sample_size / 32))
#         self.avgpool = nn.AvgPool3d(
#             (last_duration, last_size, last_size), stride=1)
#         # attention blocks
#         self.fc = nn.Linear(512 * block.expansion*4, num_classes)
#         # init the weights
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
#             elif isinstance(m, nn.BatchNorm3d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
# 
#     def _make_layer(self, block, planes, blocks, shortcut_type, stride):
#         downsample = None
#         # when the in-channel and the out-channel dismatch, downsample!!!
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             # stride once for downsample and block.
#             if shortcut_type == 'A':
#                 downsample = partial(
#                     downsample_basic_block,
#                     planes=planes * block.expansion,
#                     stride=stride)
#             else:
#                 downsample = nn.Sequential(
#                     nn.Conv3d(
#                         self.inplanes,
#                         planes * block.expansion,
#                         kernel_size=1,
#                         stride=stride,
#                         bias=False), nn.BatchNorm3d(planes * block.expansion))
# 
#         layers = []
#         # only the first block needs downsample.
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         # change inplanes for the next layer
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
# 
#         return nn.Sequential(*layers)
# 
#     def forward(self, x):
# 
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
# 
#         x = self.layer1(x)
#         # output1 = self.conv2(x)
#         output1 = self.adpool1(x).view(-1, 512)
#         x = self.layer2(x)
#         # output2 = self.conv3(x)
#         output2 = self.adpool1(x).view(-1, 512)
#         x = self.layer3(x)
#         # output3 = self.conv4(x)
#         output3 = self.adpool1(x).view(-1, 512)
#         x = self.layer4(x)
#         # output4 = self.conv5(x)
#         output4 = self.adpool1(x).view(-1, 512)
# 
#         final1 = torch.cat((output1, output2, output3, output4), dim=1)
#         final_output = self.fc(final1)
#         return final_output
# 
#     def load_my_state_dict(self, state_dict):
#         my_state_dict = self.state_dict()
#         for name, param in state_dict.items():
#             if name == 'fc.weight' or name == 'fc.bias':
#                 continue
#             my_state_dict[name].copy_(param.data)
#     def adpool1(self,x):
#         B,C,T,H,W=x.size()
#         output=nn.AdaptiveAvgPool3d((T,1,1))
#         output=output(x)
#         return output
# 
# 
# 
# model_urls = {
#     'resnet18': 'https://www.jianguoyun.com/c/dl-file/resnet-18-kinetics.pth?dt=q67aev&kv=YXF6QHpqdS5lZHUuY24&sd=a54cr&ud=B8Sbfz0nRv1pG8YNAbo0KiCnzvJHDsLYQsWjtT4b1j8&vr=1',
#     'resnet34': 'https://www.jianguoyun.com/c/dl-file/resnet-34-kinetics.pth?dt=q67acv&kv=YXF6QHpqdS5lZHUuY24&sd=a54cr&ud=BftTcvolMjyywptfxelwwjXJksCaU0ektvfMwCbMD1I&vr=1',
#     'resnet50': 'https://www.jianguoyun.com/c/dl-file/resnet-50-kinetics.pth?dt=q67atr&kv=YXF6QHpqdS5lZHUuY24&sd=a54cr&ud=uKpTbIK63qX3bHs2weOGqYYc2gtssQi-o7UqpoTaG6Q&vr=1',
#     'resnet101': '',
#     'resnet152': '',
#     'resnet200': '',
# }
# 
# 
# def resnet18(pretrained=False, **kwargs):
#     """Constructs a ResNet-18 model.
#     """
#     model = ResNet(BasicBlock, [2, 2, 2, 2], shortcut_type='A', **kwargs)
#     if pretrained:
#         checkpoint = torch.load('C:\\Users\\han006\\Desktop\\plan_A_code_v3\\pretrained_weight\\resnet-18-kinetics.pth')
#         state_dict = checkpoint['state_dict']
# 
#         new_state_dict = OrderedDict()
#         for k, v in state_dict.items():
#             name = k[7:] # remove 'module.'
#             new_state_dict[name]=v
#         model.load_my_state_dict(new_state_dict)
# 
#     return model
# 
# 
# def resnet34(pretrained=False, progress=True, **kwargs):
#     """Constructs a ResNet-34 model.
#     """
#     model = ResNet(BasicBlock, [3, 4, 6, 3], shortcut_type='A', **kwargs)
#     if pretrained:
#         checkpoint = load_state_dict_from_url(model_urls['resnet34'],
#             progress=progress)
#         state_dict = checkpoint['state_dict']
# 
#         new_state_dict = OrderedDict()
#         for k, v in state_dict.items():
#             name = k[7:] # remove 'module.'
#             new_state_dict[name]=v
#         model.load_my_state_dict(new_state_dict)
# 
#     return model
# 
# 
# def resnet50(pretrained=False, progress=True, **kwargs):
#     """Constructs a ResNet-50 model.
#     """
#     model = ResNet(Bottleneck, [3, 4, 6, 3], shortcut_type='B', **kwargs)
#     if pretrained:
#         checkpoint = load_state_dict_from_url(model_urls['resnet50'],
#             progress=progress)
#         state_dict = checkpoint['state_dict']
# 
#         new_state_dict = OrderedDict()
#         for k, v in state_dict.items():
#             name = k[7:] # remove 'module.'
#             new_state_dict[name]=v
#         model.load_my_state_dict(new_state_dict)
# 
#     return model
# 
# 
# def resnet101(pretrained=False, progress=True, **kwargs):
#     """Constructs a ResNet-101 model.
#     """
#     model = ResNet(Bottleneck, [3, 4, 23, 3], shortcut_type='B', **kwargs)
#     if pretrained:
#         checkpoint = load_state_dict_from_url(model_urls['resnet101'],
#             progress=progress)
#         state_dict = checkpoint['state_dict']
# 
#         new_state_dict = OrderedDict()
#         for k, v in state_dict.items():
#             name = k[7:] # remove 'module.'
#             new_state_dict[name]=v
#         model.load_my_state_dict(new_state_dict)
# 
#     return model
# 
# 
# def resnet152(pretrained=False, progress=True, **kwargs):
#     """Constructs a ResNet-101 model.
#     """
#     model = ResNet(Bottleneck, [3, 8, 36, 3], shortcut_type='B', **kwargs)
#     if pretrained:
#         checkpoint = load_state_dict_from_url(model_urls['resnet152'],
#             progress=progress)
#         state_dict = checkpoint['state_dict']
# 
#         new_state_dict = OrderedDict()
#         for k, v in state_dict.items():
#             name = k[7:] # remove 'module.'
#             new_state_dict[name]=v
#         model.load_my_state_dict(new_state_dict)
# 
#     return model
# 
# 
# def resnet200(pretrained=False, progress=True, **kwargs):
#     """Constructs a ResNet-101 model.
#     """
#     model = ResNet(Bottleneck, [3, 24, 36, 3], shortcut_type='B', **kwargs)
#     if pretrained:
#         checkpoint = load_state_dict_from_url(model_urls['resnet200'],
#             progress=progress)
#         state_dict = checkpoint['state_dict']
# 
#         new_state_dict = OrderedDict()
#         for k, v in state_dict.items():
#             name = k[7:] # remove 'module.'
#             new_state_dict[name]=v
#         model.load_my_state_dict(new_state_dict)
# 
#     return model
# 
# # if __name__ == '__main__':
# #     import sys
# #     sys.path.append("..")
# #     import torchvision.transforms as transforms
# #     import torch
# #     #print(torch.load('C:\\Users\\han006\\Desktop\\plan_A_code_v3\\pretrained_weight\\resnet-18-kinetics.pth'))
# #     #from dataset import CSL_Isolated
# #     sample_size = 128
# #     sample_duration = 16
# #     num_classes = 500
# #     transform = transforms.Compose([transforms.Resize([sample_size, sample_size]), transforms.ToTensor()])
# #     # dataset = CSL_Isolated(data_path="/home/haodong/Data/CSL_Isolated/color_video_125000",
# #     #     label_path="/home/haodong/Data/CSL_Isolated/dictionary.txt", frames=sample_duration,
# #     #     num_classes=num_classes, transform=transform)
# #     # cnn3d = CNN3D(sample_size=sample_size, sample_duration=sample_duration, num_classes=num_classes)
# #     data=torch.rand(1,3,16,128,128)
# #     cnn3d = resnet18(pretrained=True,sample_size=sample_size, sample_duration=sample_duration, num_classes=10)
# #     print(cnn3d(data))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.hub import load_state_dict_from_url
import torchvision
from functools import partial
from collections import OrderedDict
import math
import numpy as np
#from models.convlstm_v3 import ConvLSTM
class CNN3D(nn.Module):
    def __init__(self, sample_size=128, sample_duration=16, drop_p=0.0, hidden1=512, hidden2=256, num_classes=10):
        super(CNN3D, self).__init__()
        self.sample_size = sample_size
        self.sample_duration = sample_duration
        self.num_classes = num_classes
        # network params
        self.ch1, self.ch2, self.ch3 = 32, 48, 48
        self.k1, self.k2, self.k3 = (3, 7, 7), (3, 7, 7), (3, 5, 5)
        self.s1, self.s2, self.s3 = (2, 2, 2), (2, 2, 2), (2, 2, 2)
        self.p1, self.p2, self.p3 = (0, 0, 0), (0, 0, 0), (0, 0, 0)
        self.d1, self.d2, self.d3 = (1, 1, 1), (1, 1, 1), (1, 1, 1)
        self.hidden1, self.hidden2 = hidden1, hidden2
        self.drop_p = drop_p
        self.pool_k, self.pool_s, self.pool_p, self.pool_d = (1, 2, 2), (1, 2, 2), (0, 0, 0), (1, 1, 1)
        # Conv1
        self.conv1_output_shape = self.compute_output_shape(self.sample_duration, self.sample_size,
                                                            self.sample_size, self.k1, self.s1, self.p1, self.d1)
        # self.conv1_output_shape = self.compute_output_shape(self.conv1_output_shape[0], self.conv1_output_shape[1],
        #     self.conv1_output_shape[2], self.pool_k, self.pool_s, self.pool_p, self.pool_d)
        # Conv2
        self.conv2_output_shape = self.compute_output_shape(self.conv1_output_shape[0], self.conv1_output_shape[1],
                                                            self.conv1_output_shape[2], self.k2, self.s2, self.p2,
                                                            self.d2)
        # self.conv2_output_shape = self.compute_output_shape(self.conv2_output_shape[0], self.conv2_output_shape[1],
        #     self.conv2_output_shape[2], self.pool_k, self.pool_s, self.pool_p, self.pool_d)
        # Conv3
        self.conv3_output_shape = self.compute_output_shape(self.conv2_output_shape[0], self.conv2_output_shape[1],
                                                            self.conv2_output_shape[2], self.k3, self.s3, self.p3,
                                                            self.d3)
        # print(self.conv1_output_shape, self.conv2_output_shape, self.conv3_output_shape)

        # network architecture
        # in_channels=1 for grayscale, 3 for rgb
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=self.ch1, kernel_size=self.k1,
                               stride=self.s1, padding=self.p1, dilation=self.d1)
        self.bn1 = nn.BatchNorm3d(self.ch1)
        self.conv2 = nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2,
                               stride=self.s2, padding=self.p2, dilation=self.d2)
        self.bn2 = nn.BatchNorm3d(self.ch2)
        self.conv3 = nn.Conv3d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3,
                               stride=self.s3, padding=self.p3, dilation=self.d3)
        self.bn3 = nn.BatchNorm3d(self.ch3)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(p=self.drop_p)
        self.pool = nn.MaxPool3d(kernel_size=self.pool_k)
        self.fc1 = nn.Linear(
            self.ch3 * self.conv3_output_shape[0] * self.conv3_output_shape[1] * self.conv3_output_shape[2],
            self.hidden1)
        self.fc2 = nn.Linear(self.hidden1, self.hidden2)
        self.fc3 = nn.Linear(self.hidden2, self.num_classes)

    def forward(self, x):
        # Conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.pool(x)
        # x = self.drop(x)
        # Conv2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # x = self.pool(x)
        # x = self.drop(x)
        # Conv3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        # x = self.drop(x)
        # MLP
        #print(x.shape)
        # x.size(0) ------ batch_size
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)

        return x

    def compute_output_shape(self, D_in, H_in, W_in, k, s, p, d):
        # Conv
        D_out = np.floor((D_in + 2 * p[0] - d[0] * (k[0] - 1) - 1) / s[0] + 1).astype(int)
        H_out = np.floor((H_in + 2 * p[1] - d[1] * (k[1] - 1) - 1) / s[1] + 1).astype(int)
        W_out = np.floor((W_in + 2 * p[2] - d[2] * (k[2] - 1) - 1) / s[2] + 1).astype(int)

        return D_out, H_out, W_out


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
        #self.convlstm=ConvLSTM(input_dim=planes,hidden_dim=[planes],kernel_size=(3, 3),num_layers=1,batch_first=True,bias = True,return_all_layers = False).cuda()
        
        self.bn2 = nn.BatchNorm3d(planes)
        # self.bn3 = nn.BatchNorm3d(planes)


        # self.ca = ChannelAttention(planes)
        # self.sa = SpatialAttention()

    def forward(self, x):
        B, C, T, H, W = x.size()
        residual = x
        # conv1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # conv2
        out = self.conv2(out)
        out = self.bn2(out)

        # out =self.relu(out)
        # out=out.permute(0,2,1,3,4)
        # out = self.convlstm(out)
        # out = out[0][0].permute(0, 2, 1, 3, 4)
        # out = self.bn3(out)
        # out = self.ca(out) * out
        # out = self.sa(out) * out

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
    def __init__(self, block, layers, shortcut_type, sample_size, sample_duration, num_classes=500):
        super(ResNet, self).__init__()
        # initialize inplanes to 64, it'll be changed later
        self.inplanes = 64
        self.adpool=nn.AdaptiveAvgPool3d(1)
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
        # attention blocks
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # init the weights
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)

        l2 = self.layer2(l1)

        l3 = self.layer3(l2)

        l4 = self.layer4(l3)

        g = self.avgpool(l4)

        g = g.view(g.size(0), -1)
        x = self.fc(g)
        #return g
        # return [x, c1, c2, c3, c4]
        return x
       #  B = x.size()[0]
       #  x = self.conv1(x)
       #  x = self.bn1(x)
       #  x = self.relu(x)
       #  x = self.maxpool(x)
       #
       #  x = self.layer1(x)
       #  #output1 = self.conv2(x)
       #  output1 = self.adpool1(x).view(-1,512)
       #  x = self.layer2(x)
       # # output2 = self.conv3(x)
       #  output2 = self.adpool1(x).view(-1,512)
       #  x = self.layer3(x)
       # # output3 = self.conv4(x)
       #  output3 = self.adpool1(x).view(-1,512)
       #  x = self.layer4(x)
       # # output4 = self.conv5(x)
       #  output4 = self.adpool1(x).view(-1,512)
       #
       #  final1 = torch.cat((output1, output2, output3, output4), dim=1)
       #
       #  # final=F.relu(self.fc1(final1))
       #  # final=self.fc2(final)
       #  # final_output=self.sigmoid(final)*final1
       #  final_output = self.fc(final1)
       #  return final_output
    def load_my_state_dict(self, state_dict):
        my_state_dict = self.state_dict()
        for name, param in state_dict.items():
            if name == 'fc.weight' or name == 'fc.bias':
                continue
            my_state_dict[name].copy_(param.data)

    def adpool1(self,x):
        B,C,T,H,W=x.size()
        output=nn.AdaptiveAvgPool3d((T,1,1))
        output=output(x)
        return output


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

# if __name__ == '__main__':
#     import sys
#     sys.path.append("..")
#     import torchvision.transforms as transforms
#     import torch
#     #print(torch.load('C:\\Users\\han006\\Desktop\\plan_A_code_v3\\pretrained_weight\\resnet-18-kinetics.pth'))
#     #from dataset import CSL_Isolated
#     sample_size = 128
#     sample_duration = 16
#     num_classes = 500
#     transform = transforms.Compose([transforms.Resize([sample_size, sample_size]), transforms.ToTensor()])
#     # dataset = CSL_Isolated(data_path="/home/haodong/Data/CSL_Isolated/color_video_125000",
#     #     label_path="/home/haodong/Data/CSL_Isolated/dictionary.txt", frames=sample_duration,
#     #     num_classes=num_classes, transform=transform)
#     #cnn3d = CNN3D(sample_size=224, sample_duration=16, drop_p=0.0, hidden1=512, hidden2=256, num_classes=100).cuda()
#     data=torch.rand(1,3,16,224,224).cuda()
#     cnn3d = resnet18(pretrained=False,sample_size=sample_size, sample_duration=sample_duration, num_classes=100).cuda()
#     print(cnn3d)