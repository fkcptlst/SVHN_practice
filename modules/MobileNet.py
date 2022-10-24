import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.block1=nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(planes),
                                 nn.ReLU6()
        )
        self.block2=nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False),
                                 nn.BatchNorm2d(planes),
                                 nn.ReLU6()
        )
        self.block3=nn.Sequential(nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(out_planes),
        )

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = (out + self.shortcut(x)) if self.stride == 1 else out
        return out


class Model(nn.Module):
    # t,c,n,s
    cfg = [(6, 32, 3, 2),
           (6, 64, 4, 2),
           (6, 96, 3, 1),
           (6, 160, 3, 2)]

    # t为扩张系数，c为输出通道数，n为该层重复的次数，s为步长
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU())
        self.layers = self._make_layers(in_planes=32)
        self.hidden2 = nn.Sequential(nn.Conv2d(160, 320, kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(320),
                                     nn.ReLU())
        self._digit11 = nn.Sequential(nn.Linear(320, 10))
        self._digit21 = nn.Sequential(nn.Linear(320, 10))

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(
                    Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.size())
        x = self.hidden1(x)
        # print(x.size())
        x = self.layers(x)
        # print(x.size())
        x = self.hidden2(x)
        # print(x.size())
        x = F.avg_pool2d(x, 7)
        # print(x.size())
        x = x.view(x.size(0), -1)
        # print(x.size())

        digit1_logits = self._digit11(x)
        digit2_logits = self._digit21(x)

        return digit1_logits, digit2_logits