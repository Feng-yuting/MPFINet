import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

class FusionconBlock(nn.Module):
    def __init__(self,in_planes,c_num_heads):
        super().__init__()

        self.num_heads = c_num_heads
        self.temperature1 = nn.Parameter(torch.zeros(1))
        self.temperature2 = nn.Parameter(torch.zeros(1))
        self.conv = nn.Conv2d(in_planes*2,in_planes,1)

    def _get_channel_attn(self,x, y):
        key_x = rearrange(x, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        key_y = rearrange(y, 'b (head c) h w -> b head (h w) c', head=self.num_heads)

        key_x = torch.nn.functional.normalize(key_x, dim=-1)
        key_y = torch.nn.functional.normalize(key_y, dim=-2)

        energy = key_x @ key_y  # x(B,head,ca,cb)
        attn1 = energy.softmax(dim=-1)  # attn1(B,head,ca,cb)
        attn2 = energy.permute(0, 1, 3, 2).softmax(dim=-1)  # attn2(B,head,cb,ca)
        return attn1, attn2

    def forward(self,x,y):
        """
        inputs :
            x,y : input feature maps(B C W H)
        returns :
            x_,y_ : self x,y attention value + input feature (B C W H)
            attention: B head c c
        """
        B, C, H,W = x.shape
        attn1,attn2 = self._get_channel_attn(x,y)
        value_y = rearrange(y, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        y_att = value_y @ attn1
        y_att = rearrange(y_att, 'b head (h w) c -> b (head c) h w', h=H, w=W)
        y_ = self.temperature1*y_att+x#y_(B,C,H,W)

        value_x = rearrange(x, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        x_att = value_x @ attn2
        x_att = rearrange(x_att, 'b head (h w) c -> b (head c) h w', h=H, w=W)
        x_ = self.temperature2*x_att+y

        out = torch.concat((x_,y_),dim=1)

        return out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(4, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


if __name__ == '__main__':
    x = torch.rand(1,128,64,64)
    cs_block = CoordAtt(inp = 128, oup = 128)
    x_= cs_block(x)
    print(x_.size())
