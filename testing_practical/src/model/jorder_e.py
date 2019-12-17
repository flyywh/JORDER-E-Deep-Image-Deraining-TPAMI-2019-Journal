# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

from model import common

import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init

def make_model(args, parent=False):
    return JORDER_E(args)

class ResBlock(nn.Module):
    def __init__(self, Channels, kSize=3):
        super(ResBlock, self).__init__()
        Ch = Channels
        self.relu  = nn.ReLU()

        self.conv1 = nn.Conv2d(Ch, Ch, 3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(Ch, Ch, 3, padding=1, stride=1)

        self.conv3 = nn.Conv2d(Ch, Ch, 3, dilation=2, padding=2, stride=1)
        self.conv4 = nn.Conv2d(Ch, Ch, 3, dilation=2, padding=2, stride=1)

        self.conv5 = nn.Conv2d(Ch, Ch, 3, dilation=2, padding=2, stride=1)
        self.conv6 = nn.Conv2d(Ch, Ch, 3, dilation=4, padding=4, stride=1)

    def forward(self, x, prev_x, is_the_second):
        if is_the_second==1:
            x = x + self.relu(self.conv2(self.relu(self.conv1(x)))) + 0.1*self.relu(self.conv4(self.relu(self.conv3(x)))) + self.relu(self.conv6(self.relu(self.conv5(x))))*0.1 + prev_x
        else:
            x = x + self.relu(self.conv2(self.relu(self.conv1(x)))) + self.relu(self.conv4(self.relu(self.conv3(x))))*0.1 + self.relu(self.conv6(self.relu(self.conv5(x))))*0.1
        return x

class ResNet(nn.Module):
    def __init__(self, growRate0, nConvLayers, kSize=3):
        super(ResNet, self).__init__()
        G0 = growRate0
        C  = nConvLayers
        C  = 9

        self.convs = []

        self.res1 = ResBlock(G0)
        self.convs.append(self.res1)

        self.res2 = ResBlock(G0)
        self.convs.append(self.res2)

        self.res3 = ResBlock(G0)
        self.convs.append(self.res3)

        #self.res4 = ResBlock(G0)
        #self.convs.append(self.res4 )

        #self.res5 = ResBlock(G0)
        #self.convs.append(self.res5)

        #self.res6 = ResBlock(G0)
        #self.convs.append(self.res6)

        #self.res7 = ResBlock(G0)
        #self.convs.append(self.res7)

        #self.res8 = ResBlock(G0)
        #self.convs.append(self.res8)

        #self.res9 = ResBlock(G0)
        #self.convs.append(self.res9)

        self.C = C

    def forward(self, x, feat_pre, is_the_second):
        feat_output = []
        if is_the_second==0:
            for i in range(3):
                x = self.convs[i].forward(x, [], 0)
                feat_output.append(x)
        else:
            for i in range(3):
                x = self.convs[i].forward(x, feat_pre[i], 1)
                feat_output.append(x)

        return x, feat_output

class JORDER(nn.Module):
    def __init__(self, args):
        super(JORDER, self).__init__()
        r = args.scale[0]
        G0 = 32
        kSize = args.RDNkSize

        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]

        self.encoder = nn.Conv2d(6, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.updater = ResNet(G0, 9)

        self.mask_estimator1 = nn.Conv2d(G0, 8, kSize, padding=(kSize-1)//2, stride=1)
        self.mask_estimator2 = nn.Conv2d(8, 2, kSize, padding=(kSize-1)//2, stride=1)

        self.level_estimator1 = nn.Conv2d(G0, 8, kSize, padding=(kSize-1)//2, stride=1)
        self.level_estimator2 = nn.Conv2d(8, 3, kSize, padding=(kSize-1)//2, stride=1)

        self.mask_F_w_encoder1 = nn.Conv2d(2, 16, kSize, padding=(kSize-1)//2, stride=1)
        self.mask_F_w_encoder2 = nn.Conv2d(16, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        self.mask_F_b_encoder1 = nn.Conv2d(2, 16, kSize, padding=(kSize-1)//2, stride=1)
        self.mask_F_b_encoder2 = nn.Conv2d(16, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        self.level_F_w_encoder1 = nn.Conv2d(3, 16, kSize, padding=(kSize-1)//2, stride=1)
        self.level_F_w_encoder2 = nn.Conv2d(16, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        self.level_F_b_encoder1 = nn.Conv2d(3, 16, kSize, padding=(kSize-1)//2, stride=1)
        self.level_F_b_encoder2 = nn.Conv2d(16, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        init.constant_(self.mask_F_w_encoder2.weight, 0)
        init.constant_(self.mask_F_w_encoder2.bias, 1)
        init.constant_(self.level_F_w_encoder2.weight, 0)
        init.constant_(self.level_F_w_encoder2.bias, 1)

        self.decoder = nn.Sequential(*[
          nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1),
          nn.Conv2d(G0, 3, kSize, padding=(kSize-1)//2, stride=1)
        ])

        self.relu = nn.ReLU()

    def forward(self, x, x_prev, feat_pre, is_the_second, x_mask_prev, x_level_prev):
        x_original = x
        if is_the_second==1:
            #x_original = x_prev
            x = torch.cat([x, x_prev], 1)
        else:
            #x_original = x
            x = torch.cat([x, x], 1)

        x_F, feat_this = self.updater(self.encoder(x), feat_pre, is_the_second)

        if is_the_second==1:
            x_mask = self.mask_estimator2(self.mask_estimator1(x_F)) + x_mask_prev
            x_level = self.level_estimator2(self.level_estimator1(x_F)) + x_level_prev
        else:
            x_mask = self.mask_estimator2(self.mask_estimator1(x_F))
            x_level = self.level_estimator2(self.level_estimator1(x_F))

        x_F1 = self.mask_F_b_encoder2(self.relu(self.mask_F_b_encoder1(x_mask))) + x_F \
              + self.level_F_b_encoder2(self.relu(self.level_F_b_encoder1(x_level)))
        x_F2 = self.mask_F_w_encoder2(self.relu(self.mask_F_w_encoder1(x_mask))) * x_F \
              * self.level_F_w_encoder2(self.relu(self.level_F_w_encoder1(x_level)))

        x_combine_F = x_F1 + x_F2

        return self.decoder(x_combine_F)+x_original, feat_this, x_mask, x_level

class JORDER_E(nn.Module):
    def __init__(self, args):
        super(JORDER_E, self).__init__()

        self.jorder1 = JORDER(args)
        self.jorder2 = JORDER(args)
        # self.jorder3 = JORDER(args)

    def forward(self, x):
        x1, feat_1, x_mask1, x_level1 = self.jorder1(x, [],     [], 0, [], [])
        x2, feat_2, x_mask2, x_level2 = self.jorder2(x, x1, feat_1, 1, x_mask1, x_level1)
        # x3, feat_3 = self.jorder3(x, x2, feat_2, 1)

        return x2, x1, x_mask2, x_level2  #, x_mask1, x_level1
