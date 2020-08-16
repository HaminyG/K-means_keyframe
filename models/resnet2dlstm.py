import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

import os,inspect,sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,currentdir)

"""
Implementation of CNN+LSTM.
"""
"""
Implementation of Resnet+LSTM
"""
class ResCRNN(nn.Module):
    def __init__(self, sample_size=256, sample_duration=16, num_classes=100,
                lstm_hidden_size=512, lstm_num_layers=1, arch="resnet18",
                attention=False):
        super(ResCRNN, self).__init__()
        self.sample_size = sample_size
        self.sample_duration = sample_duration
        self.num_classes = num_classes

        # network params
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.attention = attention

        # network architecture
        if arch == "resnet18":
            resnet = models.resnet18(pretrained=False)
        elif arch == "resnet34":
            resnet = models.resnet34(pretrained=False)
        elif arch == "resnet50":
            resnet = models.resnet50(pretrained=False)
        elif arch == "resnet101":
            resnet = models.resnet101(pretrained=False)
        elif arch == "resnet152":
            resnet = models.resnet152(pretrained=False)
        # delete the last fc layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.lstm = nn.LSTM(
            input_size=resnet.fc.in_features,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
        )
        self.fc1 = nn.Linear(self.lstm_hidden_size, self.num_classes)

    def forward(self, x):
        # CNN
        cnn_embed_seq = []
        # x: (batch_size, channel, t, h, w)
        for t in range(x.size(2)):
            # with torch.no_grad():
            out = self.resnet(x[:, :, t, :, :])
            # print(out.shape)
            out = out.view(out.size(0), -1)
            cnn_embed_seq.append(out)

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0)
        # print(cnn_embed_seq.shape)
        # batch first
        cnn_embed_seq = cnn_embed_seq.transpose_(0, 1)

        # LSTM
        # use faster code paths
        self.lstm.flatten_parameters()
        out, (h_n, c_n) = self.lstm(cnn_embed_seq, None)
        # MLP
        if self.attention:
            out = self.fc1(self.attn_block(out))
        else:
            # out: (batch, seq, feature), choose the last time step
            out = self.fc1(out[:, -1, :])

        return out


# Test
if __name__ == '__main__':
    import sys
    sys.path.append("..")
    import torchvision.transforms as transforms
    sample_size = 128
    sample_duration = 16
    num_classes = 100
    # crnn = CRNN()
    data=torch.randn(1,3,16,128,128).cuda()
    crnn = ResCRNN(sample_size=sample_size, sample_duration=sample_duration, num_classes=num_classes, arch="resnet18").cuda()
    print(crnn(data).size())