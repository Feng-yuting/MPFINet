import torch
import torch.nn as nn
from Attention import CoordAtt
from dynamic_conv import Dynamic_conv2d
from config import opt

class one_conv(nn.Module):
    def __init__(self,inchanels,growth_rate):
        super(one_conv,self).__init__()
        self.conv0 = nn.Sequential(
            Dynamic_conv2d(in_planes=inchanels, out_planes=growth_rate, kernel_size=1, ratio=0.25, padding=0),
            nn.ReLU(growth_rate),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(inchanels, growth_rate,kernel_size=1,padding = 0,stride= 1),
            nn.ReLU(growth_rate),
            Dynamic_conv2d(in_planes=growth_rate, out_planes=growth_rate, kernel_size=1, ratio=0.25, padding=0),
            nn.ReLU(growth_rate),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inchanels, growth_rate, kernel_size=3, padding=1, stride=1),
            nn.ReLU(growth_rate),
            Dynamic_conv2d(in_planes=growth_rate, out_planes=growth_rate, kernel_size=3, ratio=0.25, padding=1),
            nn.ReLU(growth_rate),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(inchanels, growth_rate, kernel_size=1, padding=0, stride=1),
            nn.ReLU(growth_rate),
            CoordAtt(growth_rate,growth_rate),
        )

    def forward(self,x):
        branch0 = self.conv0(x)
        branch1 = self.conv1(x)
        branch2 = self.conv2(x)
        branch3 = self.conv3(x)
        output = torch.cat((branch0,branch1, branch2,branch3,x), dim=1)

        return output

class MDCDB(nn.Module):
    def __init__(self,G0,G_out,C,G,drop = 0.3):
        super(MDCDB,self).__init__()
        convs = []
        for i in range(C):
            convs.append(one_conv(G0+4*i*G,G))
        self.conv = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(G0+4*C*G,G_out,kernel_size = 1,padding = 0,stride =1)
        self.dropout = nn.Dropout(drop) if drop > 0. else nn.Identity()


    def forward(self,x):
        out = self.conv(x)
        lff = self.dropout(self.LFF(out))
        return lff+ x





if __name__ == '__main__':
    input= torch.rand(1, 16, 256, 256)
    D = 2 #整个模块的数量
    RDBS = []
    for d in range(D):
        RDBS.append(RDB(G0 = 16, G_out = 16,C = 3, G = 8))
    RDBS = nn.Sequential(*RDBS)

    output = RDBS(input)
    print('output输出',output.size())
