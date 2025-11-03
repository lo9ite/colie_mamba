import torch
import torch.nn as nn

try:
    from model_mamba_lowlight import SS2D
except Exception as e:
    raise ImportError("无法导入 SS2D，请确保 model_mamba_lowlight.py 存在并包含该类。")

class MambaBlock(nn.Module):
    """多方向 SS2D 融合版 MambaBlock（修正垂直扫描）"""
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.mamba_h = SS2D(d_model, d_state, d_conv, expand, dropout=dropout)
        self.mamba_h_inv = SS2D(d_model, d_state, d_conv, expand, dropout=dropout)
        self.mamba_v = SS2D(d_model, d_state, d_conv, expand, dropout=dropout)
        self.mamba_v_inv = SS2D(d_model, d_state, d_conv, expand, dropout=dropout)

        self.fuse = nn.Conv2d(4 * d_model, d_model, kernel_size=1, bias=True)
        self.norm = nn.LayerNorm(4 * d_model)
        self.act = nn.SiLU()
        self.res_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % 4 == 0 and W % 4 == 0, f"输入尺寸必须为4的倍数 (got {H}x{W})"

        x_in = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)

        # 水平扫描
        y_h = self.mamba_h(x_in)
        # 水平反向扫描
        y_h_inv = torch.flip(self.mamba_h_inv(torch.flip(x_in, dims=[2])), dims=[2])

        # 垂直扫描：转置 H/W，让 SS2D 沿竖向“看”
        y_v = self.mamba_v(x_in.transpose(1, 2))
        y_v = y_v.transpose(1, 2).contiguous()

        # 垂直反向扫描
        y_v_inv = torch.flip(self.mamba_v_inv(torch.flip(x_in.transpose(1, 2), dims=[2])), dims=[2])
        y_v_inv = y_v_inv.transpose(1, 2).contiguous()

        # 合并
        y_cat = torch.cat([y_h, y_h_inv, y_v, y_v_inv], dim=-1)  # (B,H,W,4C)
        y_cat = self.norm(y_cat)
        y_cat = self.act(y_cat)
        y_cat = y_cat.permute(0, 3, 1, 2).contiguous()  # (B,4C,H,W)

        # 融合 + 残差
        y_out = self.fuse(y_cat)
        return x + self.res_scale * y_out
