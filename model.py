import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F
import numpy as np
from utils import Downsample,Upsample,SimpleGate,get_edge
from IRB import CSTB
from Attention import FusionconBlock
from FEB import MDCDB
from config import opt
from initalize import init_weights

class MPFINet(nn.Module):
    def __init__(self):
        super(MPFINet, self).__init__()

        # define head body
        self.head_ms = nn.Conv2d(in_channels=3,
                                 out_channels=opt.stem_64,
                                 kernel_size=opt.kernel_size,
                                 padding=(opt.kernel_size // 2),
                                 stride=1)
        self.head_pan = nn.Conv2d(in_channels=3,
                                  out_channels=opt.stem_16,
                                  kernel_size=opt.kernel_size,
                                  padding=(opt.kernel_size // 2),
                                  stride=1)
        D = 3
        Cnum = 2
        RDBS1 = []
        for d in range(D):
            RDBS1.append(MDCDB(G0=opt.stem_16,G_out=opt.stem_16, C=Cnum, G=opt.stem_16//2))
        self.RDBS1 = nn.Sequential(*RDBS1)

        #stage2 downsample
        self.Downsample1 = Downsample(n_feat=opt.stem_16)

        RDBS2 = []
        for d in range(D):
            RDBS2.append(MDCDB(G0=opt.stem_32,G_out=opt.stem_32, C=Cnum, G=opt.stem_32//2))
        self.RDBS2 = nn.Sequential(*RDBS2)

        #stage3 Downsample
        self.Downsample2 = Downsample(n_feat=opt.stem_32)
        RDBS3 = []
        for d in range(D):
            RDBS3.append(MDCDB(G0=opt.stem_64, G_out=opt.stem_64, C=Cnum, G=opt.stem_64 // 2))
        self.RDBS3 = nn.Sequential(*RDBS3)


        # ################Fusion
        self.fusion1 = FusionconBlock(in_planes = 64,c_num_heads = 8)
        self.fusion2 = FusionconBlock(in_planes = 32,c_num_heads = 4)
        self.fusion3 = FusionconBlock(in_planes = 16,c_num_heads = 2)

        ################MS
        self.tranf1_ms = nn.Sequential(*[
            CSTB(ms_stem=opt.stem_64*2, expansion=opt.expansion,num_heads=opt.channel_heads[0])
                                        for i in range(3)])
        self.sg = SimpleGate()
        self.Upsample1 = Upsample(opt.stem_64)

        #stage2
        self.tranf2_ms = nn.Sequential(*[
            CSTB(ms_stem=opt.stem_64, expansion=opt.expansion,num_heads=opt.channel_heads[1])
                                        for i in range(3)])
        self.Upsample2 = Upsample(opt.stem_32)

        #stage3
        self.tranf3_ms =  nn.Sequential(*[
            CSTB(ms_stem=opt.stem_32, expansion=opt.expansion,num_heads=opt.channel_heads[2])
                                        for i in range(3)])
        # tail
        self.tail = nn.Sequential(
            nn.Conv2d(in_channels=opt.stem_16, out_channels=3, kernel_size=opt.kernel_size,
                      padding=(opt.kernel_size // 2), stride=1),
            nn.PReLU()
        )

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward_once(self, xr4):
        # shallow feature extract
        xr4_1 = self.head_pan(xr4)#(B,16,256,256)
        #stage1
        xr4_cnn1 = self.RDBS1(xr4_1)# (B,16,256,256)
        xr4_1_out = xr4_1+xr4_cnn1

        #stage2
        xr4_2 = self.Downsample1(xr4_1_out)# (B,32,128,128)
        xr4_cnn2 = self.RDBS2(xr4_2)  # (B,32,128,128)
        xr4_2_out = xr4_2+ xr4_cnn2

        #stage3
        xr4_3 = self.Downsample2(xr4_2_out)  # (B,64,64,64)
        xr4_cnn3 = self.RDBS3(xr4_3)  # (B,64,64,64)
        xr4_3_out = xr4_3+xr4_cnn3

        return xr4_1_out,xr4_2_out,xr4_3_out#x1,x2,x3

    def forward(self, x_pan, x_ms):
        x_pan = torch.concat((x_pan,x_pan,x_pan),dim=1)
        x_ms_up4 = F.interpolate(x_ms, scale_factor=4, mode='bicubic', align_corners=False, recompute_scale_factor=True)
        x_residue4 = x_pan - x_ms_up4
        re_pan1, re_pan2, re_pan3 = self.forward_once(x_residue4)

        ############MS
        x_ms = self.head_ms(x_ms)  # ms(64*64*64)
        mp_fused1 = self.fusion1(x_ms,re_pan3)# (B,128,64,64)
        mp_tran1 = self.sg(self.tranf1_ms(mp_fused1))# (B,64,64,64)
        mp_tran1 = mp_tran1 + re_pan3 + x_ms
        mp_up1 = self.Upsample1(mp_tran1)# (B,32,128,128)

        #stage2
        mp_fused2 = self.fusion2(mp_up1, re_pan2) # (B,64,128,128)
        mp_tran2 = self.sg(self.tranf2_ms(mp_fused2))  # (B,32,128,128)
        mp_tran2 = mp_tran2 + re_pan2 + mp_up1
        mp_up2 = self.Upsample2(mp_tran2)  # (B,16,256,256)

        # stage3
        mp_fused3 = self.fusion3(mp_up2, re_pan1)# (B,32,256,256)
        mp_tran3 = self.sg(self.tranf3_ms(mp_fused3))  # (B,16,256,256)
        mp_tran3 = mp_tran3 + re_pan1 + mp_up2

        #tail
        x = self.tail(mp_tran3)

        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


if __name__ == "__main__":
    model = MPFINet().cuda()
    model.train()
    input_ms,input_pan = torch.rand(1, 3, 64, 64),torch.rand(1,1,256,256)
    sr= model(input_pan.cuda(),input_ms.cuda())
    print('size',sr.size())
