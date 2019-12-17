# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

from model import common

import torch
import torch.nn as nn


def make_model(args, parent=False):
    return RDN(args)

class RDB_Conv(nn.Module):
    def __init__(self, Channels, kSize=3):
        super(RDB_Conv, self).__init__()
        Ch = Channels
        self.conv1 = nn.Conv2d(Ch, Ch, kSize, padding=(kSize-1)//2, stride=1)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(Ch, Ch, kSize, padding=(kSize-1)//2, stride=1)

    def forward(self, x):
        return x + self.relu(self.conv2(self.relu(self.conv1(x))))

class RDB(nn.Module):
    def __init__(self, growRate0, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        C  = nConvLayers
        
        self.conv1 = RDB_Conv(G0)
        self.conv2 = RDB_Conv(G0)
        self.conv3 = RDB_Conv(G0)
        self.conv4 = RDB_Conv(G0)
        
        self.LFF = nn.Conv2d(G0+16*7, G0, 1, padding=0, stride=1)
        self.C = C

    def forward(self, x):
        res = []
        ox = x

        x = self.conv1(x)
        res.append(torch.narrow(x, 1, 0, 16))
        x = self.conv2(x)
        res.append(torch.narrow(x, 1, 0, 16))
        x = self.conv3(x)
        res.append(torch.narrow(x, 1, 0, 16))
        x = self.conv4(x)
        res.append(x)


        return self.LFF(torch.cat(res,1)) + ox

class RDB_L2(nn.Module):
    def __init__(self, growRate0, nConvLayers, kSize=3):
        super(RDB_L2, self).__init__()
        G0 = growRate0
        C  = nConvLayers

        self.conv1 = RDB(growRate0 = G0, nConvLayers = C)
        self.conv2 = RDB(growRate0 = G0, nConvLayers = C)

        self.LFF = nn.Conv2d(G0*4, G0, 1, padding=0, stride=1)
        self.C = C

    def forward(self, x):
        res = []

        ox = x
        x = self.conv1(x)
	res.append(torch.narrow(x, 1, 0, 16))

        x = self.conv2(x)
        res.append(torch.narrow(x, 1, 0, 16)) 

        x = self.conv3(x)
        res.append(torch.narrow(x, 1, 0, 16)) 

        x = self.conv4(x)
        res.append(x)

        return self.LFF(torch.cat(res,1)) + ox

class RDN(nn.Module):
    def __init__(self, args):
        super(RDN, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize

        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]
 
        self.D = 4 

        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        self.RDBs = nn.ModuleList()
        for i in range(3):
            self.RDBs.append(
                RDB_L2(growRate0 = G0, nConvLayers = C)
            )

        self.GFF = nn.Sequential(*[
            nn.Conv2d(3 * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        if r==2:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G, kSize, padding=(kSize-1)//2, stride=1),
                nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ]) 

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(3):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out,1))
        x += f__1

        return self.UPNet(x)
