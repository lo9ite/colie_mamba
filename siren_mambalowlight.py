import torch
import torch.nn as nn
import torch.nn.functional as F
from mambalowlight import MambaBlock


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.mamba = MambaBlock(in_channels, d_state, d_conv, expand)
        self.act = nn.SiLU()
        self.down = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x):
        x = self.mamba(x)
        skip = self.act(x)
        x = self.down(skip)
        return x, skip


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, skip_channels, 2, stride=2)
        self.mamba = MambaBlock(skip_channels * 2, d_state, d_conv, expand)
        self.conv_out = nn.Conv2d(skip_channels * 2, out_channels, 1)
        self.act = nn.SiLU()

    def forward(self, x, skip):
        x = self.up(x)
        # 对齐大小
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.mamba(x)
        x = self.act(x)
        return self.conv_out(x)


class DenoiseNet(nn.Module):
    """轻量卷积降噪网络"""
    def __init__(self, channels=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, 3, padding=1)
        )

    def forward(self, x):
        return torch.clamp(x - self.block(x), 0, 1)


class INF_MAMBA(nn.Module):
    """UNet + 多方向Mamba + 降噪"""
    def __init__(self, in_channels=3, base_channels=32, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.SiLU()
        )
        # 128 -> 64 -> 32
        self.enc1 = EncoderBlock(base_channels, base_channels * 2, d_state, d_conv, expand)
        self.enc2 = EncoderBlock(base_channels * 2, base_channels * 4, d_state, d_conv, expand)
        # bottleneck
        self.bottleneck = MambaBlock(base_channels * 4, d_state, d_conv, expand)
        # 32 -> 64 -> 128
        self.dec1 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2, d_state, d_conv, expand)
        self.dec2 = DecoderBlock(base_channels * 2, base_channels, base_channels, d_state, d_conv, expand)
        self.output_conv = nn.Conv2d(base_channels, 1, 3, padding=1)
        #self.denoise = DenoiseNet(1)

    def get_coord(self, H, W, device):
        y = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1).repeat(1, 1, 1, W)
        x = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W).repeat(1, 1, H, 1)
        return torch.cat([x, y], dim=1)

    def forward(self, img_v):
        B, _, H, W = img_v.shape
        coord = self.get_coord(H, W, img_v.device).repeat(B, 1, 1, 1)
        x = torch.cat([img_v, coord], dim=1)
        x = self.input(x)
        x1, skip1 = self.enc1(x)  # 128 -> 64
        x2, skip2 = self.enc2(x1) # 64 -> 32
        x_b = self.bottleneck(x2)
        x = self.dec1(x_b, skip2)
        x = self.dec2(x, skip1)
        illu = torch.sigmoid(self.output_conv(x))
        #return self.denoise(illu)
        return illu
