import torch
import torch.nn as nn
from torch.utils.data import Dataset

class UNet2D(nn.Module):
    def __init__(self, in_ch=6, out_ch=6, wf=4):
        super().__init__()
        # down1: in_ch → 2**wf      = 16
        # down2: 2**wf → 2**(wf+1)   = 32
        # bottleneck:           2**(wf+2) = 64
        self.down1 = nn.Sequential(
            nn.Conv2d(in_ch, 2**wf, 3, padding=1), nn.ReLU(),
            nn.Conv2d(2**wf, 2**wf, 3, padding=1), nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = nn.Sequential(
            nn.Conv2d(2**wf, 2**(wf+1), 3, padding=1), nn.ReLU(),
            nn.Conv2d(2**(wf+1), 2**(wf+1), 3, padding=1), nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(2**(wf+1), 2**(wf+2), 3, padding=1), nn.ReLU(),
            nn.Conv2d(2**(wf+2), 2**(wf+2), 3, padding=1), nn.ReLU(),
        )

        # upsample + concat + double conv
        self.up2 = nn.ConvTranspose2d(2**(wf+2), 2**(wf+1), 2, stride=2)
        self.upconv2 = nn.Sequential(
            nn.Conv2d(2**(wf+2), 2**(wf+1), 3, padding=1), nn.ReLU(),
            nn.Conv2d(2**(wf+1), 2**(wf+1), 3, padding=1), nn.ReLU(),
        )
        self.up1 = nn.ConvTranspose2d(2**(wf+1), 2**wf, 2, stride=2)
        self.upconv1 = nn.Sequential(
            nn.Conv2d(2**(wf+1), 2**wf, 3, padding=1), nn.ReLU(),
            nn.Conv2d(2**wf, 2**wf, 3, padding=1), nn.ReLU(),
        )

        # final conv
        self.final = nn.Conv2d(2**wf, out_ch, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        b = self.bottleneck(p2)

        u2 = self.up2(b)
        c2 = torch.cat([u2, d2], dim=1)
        uc2 = self.upconv2(c2)

        u1 = self.up1(uc2)
        c1 = torch.cat([u1, d1], dim=1)
        uc1 = self.upconv1(c1)

        return self.final(uc1)


class WeightNet(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nf, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(32, nf)

    def forward(self, x):
        x = self.conv(x).squeeze(-1).squeeze(-1)
        return torch.softmax(self.fc(x), dim=-1)
