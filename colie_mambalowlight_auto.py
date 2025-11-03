from utils import *
from loss import *
from siren_mambalowlight import INF_MAMBA
from color import rgb2hsv_torch, hsv2rgb_torch
import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='CoLIE + Multi-Direction Mamba + Adaptive Exposure Control')
parser.add_argument('--input_folder', type=str, default='input/dataset/LOLdataset/eval15/low')
parser.add_argument('--output_folder', type=str, default='output/dataset/LOLdataset/eval15/mamba_adaptive')
parser.add_argument('--down_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=800)
parser.add_argument('--base_channels', type=int, default=32)
# 初始超参数
parser.add_argument('--alpha', type=float, default=0.08)
parser.add_argument('--beta', type=float, default=0.3)
parser.add_argument('--gamma', type=float, default=0.4)
parser.add_argument('--delta', type=float, default=1.2)
parser.add_argument('--L', type=float, default=0.45)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=3e-4)
opt = parser.parse_args()

os.makedirs(opt.output_folder, exist_ok=True)

# 亮度调节参数
ADAPT_STEP = 0.05     # 每次调整步幅 ±5%
ADAPT_TOL = 0.05      # 亮度容忍范围 ±0.05 around L
PRINT_INTERVAL = 50   # 输出间隔

for img_name in tqdm(np.sort(os.listdir(opt.input_folder))):
    img_path = os.path.join(opt.input_folder, img_name)
    img_rgb = get_image(img_path)
    img_hsv = rgb2hsv_torch(img_rgb)
    img_v = get_v_component(img_hsv)
    H_org, W_org = img_v.shape[2:]

    img_v_lr = interpolate_image(img_v, opt.down_size, opt.down_size)

    # 初始化模型与优化器
    model = INF_MAMBA(in_channels=3, base_channels=opt.base_channels).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    # 损失函数
    l_exp, l_TV = L_exp(16, opt.L), L_TV()

    # 当前动态权重
    alpha, beta, gamma, delta, L_target = opt.alpha, opt.beta, opt.gamma, opt.delta, opt.L

    model.train()
    for epoch in range(opt.epochs):
        optimizer.zero_grad()
        illu_res_lr = model(img_v_lr)
        illu_lr = illu_res_lr + img_v_lr
        img_v_fixed_lr = img_v_lr / (illu_lr + 1e-4)

        # 各损失
        loss_fidelity = F.mse_loss(illu_lr, img_v_lr)
        loss_tv = l_TV(illu_lr)
        loss_exp = l_exp(illu_lr)
        loss_sparsity = torch.mean(img_v_fixed_lr)

        total_loss = (
            delta * loss_fidelity +
            beta * loss_tv +
            gamma * loss_exp +
            alpha * loss_sparsity
        )
        total_loss.backward()
        optimizer.step()

        # ---------- 🔍 Adaptive Exposure Balancing ----------
        with torch.no_grad():
            mean_lum = illu_lr.mean().item()
            diff = mean_lum - L_target

            if abs(diff) > ADAPT_TOL:
                if diff > 0:  # 过曝 → 降γ,升α
                    gamma *= (1 - ADAPT_STEP)
                    alpha *= (1 + ADAPT_STEP)
                else:          # 欠曝 → 升γ,降α
                    gamma *= (1 + ADAPT_STEP)
                    alpha *= (1 - ADAPT_STEP)

                # 限制范围防止失控
                gamma = float(np.clip(gamma, 0.1, 1.2))
                alpha = float(np.clip(alpha, 0.02, 0.2))

        # ---------- 日志输出 ----------
        if (epoch + 1) % PRINT_INTERVAL == 0:
            tqdm.write(
                f"{img_name} | Epoch {epoch+1}/{opt.epochs} | "
                f"Loss: {total_loss.item():.4f} | "
                f"Lum={mean_lum:.3f} | α={alpha:.3f}, γ={gamma:.3f}"
            )

    # ---------- 推理 & 保存 ----------
    img_v_fixed = filter_up(img_v_lr, img_v_fixed_lr, img_v, r=1)
    img_hsv_fixed = replace_v_component(img_hsv, img_v_fixed)
    img_rgb_fixed = hsv2rgb_torch(img_hsv_fixed)
    img_rgb_fixed = torch.clamp(img_rgb_fixed / torch.max(img_rgb_fixed), 0, 1)

    # gamma校正（防止微过曝）
    img_rgb_fixed = torch.pow(img_rgb_fixed, 0.9)

    img_np = (torch.movedim(img_rgb_fixed, 1, -1)[0].cpu().detach().numpy() * 255).astype(np.uint8)
    Image.fromarray(img_np).save(os.path.join(opt.output_folder, img_name))

print("\n✅ 所有图像处理完成（已启用自适应曝光控制）")
