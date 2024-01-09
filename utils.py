import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from config import *
from initalize import init_weights
import cv2
import numpy as np
from config import *

##########################################################################
## Convolution on MS
class ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes,droprate):
        super(ConvBlock, self).__init__()


        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.act1 = nn.PReLU()
        #Dropout
        self.drop = nn.Dropout(droprate)

        self.conv2 = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.act2 = nn.PReLU()

        # self.conv3 = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn3 = nn.BatchNorm2d(outplanes)
        # self.act3 = nn.ReLU(inplace=True)

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x2 = self.act2(x)

        out = x2 + residual

        return out


class CNN2Trans(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride):
        super(CNN2Trans, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = nn.LayerNorm(outplanes,eps=1e-6)
        self.act = nn.GELU()

    def forward(self, x, x_tran):
        x = self.conv_project(x)  # [N, C, H, W]

        x = self.sample_pooling(x).flatten(2).transpose(1, 2)#[N,C,H_new*W_new]----[N,H_new*W_new,C]
        x = self.ln(x)
        x = self.act(x)

        x = x_tran+ x

        return x
##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

##########################################################################
## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

##########################################################################
## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, in_planes, reduction, bn,act):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(
                nn.Conv2d(in_planes,in_planes,kernel_size=3,padding=1,stride=1)
            )
            if bn: modules_body.append(nn.BatchNorm2d(in_planes))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(in_planes, reduction))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

# get high-frequency
def get_edge(data):
    data = data.cpu().numpy()
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape) == 3:
            rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5),
                                                        normalize=True)  # 第二个参数的-1表示输出图像使用的深度与输入图像相同
        else:
            rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5), normalize=True)
    return torch.from_numpy(rs).to(opt.device)

if __name__ == '__main__':
    x = torch.rand(1,32,64,64)
    up = Downsample(n_feat=32)
    out = up(x)
    print(out.size())