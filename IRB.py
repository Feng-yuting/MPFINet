import torch
import torch.nn as nn
from einops import rearrange
from config import opt


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class V_MaskAttention(nn.Module):
    def __init__(self,input_channel):
        super(V_MaskAttention, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=1, padding=0, bias=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=5, padding=2, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        x2 = self.sigmoid(self.conv2(x))
        att = x+(x * x2)
        return att


class AttentionMs(nn.Module):
    def __init__(self, dim,num_heads,drop):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.dropout = nn.Dropout(drop) if drop > 0. else nn.Identity()
        self.maskattention = V_MaskAttention(input_channel = dim)
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)
        v = self.maskattention(x)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)


        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)

        return out



class Mlp(nn.Module):
    def __init__(self,in_features,expansion,drop_out_rate):
        super().__init__()
        hidden_feature = in_features*expansion
        self.mlp = nn.Sequential(
            nn.Conv2d(in_features,hidden_feature,1,1,0),
            nn.GELU(),
            nn.Conv2d(hidden_feature, in_features, 1, 1, 0),
        )
        self.dropout = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
    def forward(self,x):
        res = x
        x = self.dropout(self.mlp(x))
        x = x + res
        return x#(B,C,H,W)



class CSTB(nn.Module):

    def __init__(self,ms_stem,expansion, num_heads,drop=0.3):
        super().__init__()

        self.att = AttentionMs(ms_stem,num_heads,drop)
        self.mlp = Mlp(in_features=ms_stem, expansion=expansion,drop_out_rate = drop)
        self.norm2d = LayerNorm2d(ms_stem)

    def forward(self, x):#(B,C,H,W)

        residual1 = x
        x = self.norm2d(x)
        x = self.att(x)
        x = residual1+x

        residual2 = x
        x = self.norm2d(x)
        x = self.mlp(x)
        x = residual2+x

        return x#(Batch_size,64,64,64)





