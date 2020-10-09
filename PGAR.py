import torch
import torch.nn as nn
import torch.nn.functional as F
from VGG16 import VGG

class Convert(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Convert, self).__init__()
        self.conv1 = nn.Conv2d(in_channel[0], out_channel[0], 1)
        self.conv2 = nn.Conv2d(in_channel[1], out_channel[1], 1)
        self.conv3 = nn.Conv2d(in_channel[2], out_channel[2], 1)
        self.conv4 = nn.Conv2d(in_channel[3], out_channel[3], 1)
        self.conv5 = nn.Conv2d(in_channel[4], out_channel[4], 1)

    def forward(self, x1, x2, x3, x4, x5):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)
        x5 = self.conv5(x5)

        return x1, x2, x3, x4, x5

class DEP(nn.Module):
    def __init__(self, out_channel):
        super(DEP, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, out_channel, 3, padding=1, stride=2), nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, stride=2), nn.ReLU(True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, stride=2), nn.ReLU(True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, stride=2), nn.ReLU(True),
        )

    def forward(self, d):
        d = self.conv1(d)
        d3 = self.conv2(d)
        d4 = self.conv3(d3)
        d5 = self.conv4(d4)

        return d3, d4, d5

class BasicResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, rate):
        super(BasicResBlock, self).__init__()
        self.squeeze = nn.Conv2d(in_channel, out_channel, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=rate, dilation=rate), nn.ReLU(True),
        )
        self.expand = nn.Conv2d(out_channel, in_channel, 1)

    def forward(self, x):
        x0 = self.squeeze(x)
        x0 = self.conv(x0)
        x = x + self.expand(x0)

        return x

class MSR(nn.Module):
    def __init__(self, in_channel, out_channel, num):
        super(MSR, self).__init__()
        self.BasicRes1 = BasicResBlock(in_channel, out_channel, 1)
        self.BasicRes2 = BasicResBlock(in_channel, out_channel, 2)
        self.BasicRes3 = BasicResBlock(in_channel, out_channel, 3)
        self.score = nn.Conv2d(in_channel, 1, 3, padding=1)
        self.num = num

    def forward(self, x):
        x1 = self.BasicRes1(x)
        for n in range(1, self.num):
            x1 = self.BasicRes1(x1)

        x2 = self.BasicRes2(x)
        for n in range(1, self.num):
            x2 = self.BasicRes2(x2)

        x3 = self.BasicRes3(x)
        for n in range(1, self.num):
            x3 = self.BasicRes3(x3)

        x = x1 + x2 + x3
        x = self.score(x)

        return x

class GuidedResBlock(nn.Module):
    def __init__(self, channel, subchannel):
        super(GuidedResBlock, self).__init__()
        self.group = channel//subchannel
        self.conv = nn.Sequential(
            nn.Conv2d(channel + self.group, channel, 3, padding=1), nn.ReLU(True),
        )
        self.score = nn.Conv2d(channel, 1, 3, padding=1)

    def forward(self, x, y):
        if self.group == 1:
            x_cat = torch.cat((x, y), 1)
        elif self.group == 2:
            xs = torch.chunk(x, 2, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y), 1)
        elif self.group == 4:
            xs = torch.chunk(x, 4, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y), 1)
        elif self.group == 8:
            xs = torch.chunk(x, 8, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y), 1)
        elif self.group == 16:
            xs = torch.chunk(x, 16, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
            xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y), 1)
        elif self.group == 32:
            xs = torch.chunk(x, 32, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
            xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y,
            xs[16], y, xs[17], y, xs[18], y, xs[19], y, xs[20], y, xs[21], y, xs[22], y, xs[23], y,
            xs[24], y, xs[25], y, xs[26], y, xs[27], y, xs[28], y, xs[29], y, xs[30], y, xs[31], y), 1)
        else:
            xs = torch.chunk(x, 64, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
            xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y,
            xs[16], y, xs[17], y, xs[18], y, xs[19], y, xs[20], y, xs[21], y, xs[22], y, xs[23], y,
            xs[24], y, xs[25], y, xs[26], y, xs[27], y, xs[28], y, xs[29], y, xs[30], y, xs[31], y,
            xs[32], y, xs[33], y, xs[34], y, xs[35], y, xs[36], y, xs[37], y, xs[38], y, xs[39], y,
            xs[40], y, xs[41], y, xs[42], y, xs[43], y, xs[44], y, xs[45], y, xs[46], y, xs[47], y,
            xs[48], y, xs[49], y, xs[50], y, xs[51], y, xs[52], y, xs[53], y, xs[54], y, xs[55], y,
            xs[56], y, xs[57], y, xs[58], y, xs[59], y, xs[60], y, xs[61], y, xs[62], y, xs[63], y), 1)

        x = x + self.conv(x_cat)
        y = y + self.score(x)

        return x, y

class GR(nn.Module):
    def __init__(self, channel):
        super(GR, self).__init__()
        self.weak_gr = GuidedResBlock(channel, channel)
        self.medium_gr = GuidedResBlock(channel, 8)
        self.strong_gr = GuidedResBlock(channel, 1)

    def forward(self, x, y):
        x, y = self.weak_gr(x, y)
        x, y = self.medium_gr(x, y)
        _, y = self.strong_gr(x, y)

        return y

class PGAR(nn.Module):
    def __init__(self, channel=64):
        super(PGAR, self).__init__()
        self.vgg = VGG()
        self.dep = DEP(channel)
        self.convert = Convert([64, 128, 256, 512, 512], [16, 32, 64, 64, 64])
        self.msr = MSR(512, channel, 5)
        self.gr1 = GR(channel//4)
        self.gr2 = GR(channel//2)
        self.gr3 = GR(channel)
        self.gr4 = GR(channel)
        self.gr5 = GR(channel)
        self.grd3 = GR(channel)
        self.grd4 = GR(channel)
        self.grd5 = GR(channel)

    def forward(self, x, d):
        x1 = self.vgg.conv1(x)
        x2 = self.vgg.conv2(x1)
        x3 = self.vgg.conv3(x2)
        x4 = self.vgg.conv4(x3)
        x5 = self.vgg.conv5(x4)
        x6 = self.vgg.conv6(x5)

        d3, d4, d5 = self.dep(d)
        x1, x2, x3, x4, x5 = self.convert(x1, x2, x3, x4, x5)

        x_size = x.size()[2:]
        x2_size = x2.size()[2:]
        x3_size = x3.size()[2:]
        x4_size = x4.size()[2:]
        x5_size = x5.size()[2:]

        y6 = self.msr(x6)
        score6 = F.interpolate(y6, x_size, mode='bilinear', align_corners=True)

        y6_5 = F.interpolate(y6, x5_size, mode='bilinear', align_corners=True)
        y5a = self.grd5(d5, y6_5)
        score5a = F.interpolate(y5a, x_size, mode='bilinear', align_corners=True)

        y5b = self.gr5(x5, y5a)
        score5b = F.interpolate(y5b, x_size, mode='bilinear', align_corners=True)

        y5_4 = F.interpolate(y5b, x4_size, mode='bilinear', align_corners=True)
        y4a = self.grd4(d4, y5_4)
        score4a = F.interpolate(y4a, x_size, mode='bilinear', align_corners=True)

        y4b = self.gr4(x4, y4a)
        score4b = F.interpolate(y4b, x_size, mode='bilinear', align_corners=True)

        y4_3 = F.interpolate(y4b, x3_size, mode='bilinear', align_corners=True)	
        y3a = self.grd3(d3, y4_3)
        score3a = F.interpolate(y3a, x_size, mode='bilinear', align_corners=True)

        y3b = self.gr3(x3, y3a)
        score3b = F.interpolate(y3b, x_size, mode='bilinear', align_corners=True)

        y3_2 = F.interpolate(y3b, x2_size, mode='bilinear', align_corners=True)
        y2 = self.gr2(x2, y3_2)
        score2 = F.interpolate(y2, x_size, mode='bilinear', align_corners=True)
		
        score1 = self.gr1(x1, score2)

        return score1, score2, score3a, score3b, score4a, score4b, score5a, score5b, score6
