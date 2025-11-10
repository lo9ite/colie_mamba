# ============================================================
# colie_mambalowlight_supervised.py
# 有监督低光增强 + SASWLoss（训练+验证）+ 最优模型保存
# 兼容 LOLdataset: our485(训练) + eval15(验证)
# ============================================================

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
from losses import SASWLoss


# ----------------------------
# 参数设置
# ----------------------------
parser = argparse.ArgumentParser(description='CoLIE with Multi-Direction Mamba (Supervised + SASW + Validation)')
parser.add_argument('--input_folder', type=str, default='input/dataset/LOLdataset/our485',
                    help='训练集主目录 (包含 low, high 子目录)')
parser.add_argument('--val_folder', type=str, default='input/dataset/LOLdataset/eval15',
                    help='验证集主目录 (包含 low, high 子目录)')
parser.add_argument('--output_folder', type=str, default='output/dataset/LOLdataset/train/mamba_supervised')
parser.add_argument('--down_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=800)
parser.add_argument('--base_channels', type=int, default=32)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--beta', type=float, default=20)
parser.add_argument('--gamma', type=float, default=8)
parser.add_argument('--delta', type=float, default=5)
parser.add_argument('--L', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=3e-4)
parser.add_argument('--lambda_sup', type=float, default=1.0, help='V 通道监督损失权重')
parser.add_argument('--lambda_sasw', type=float, default=0.5, help='SASWLoss 权重')
parser.add_argument('--val_interval', type=int, default=100, help='每隔多少 epoch 在 eval15 上验证一次')
opt = parser.parse_args()

os.makedirs(opt.output_folder, exist_ok=True)

# ----------------------------
# 数据集路径
# ----------------------------
train_low_dir = os.path.join(opt.input_folder, 'low')
train_high_dir = os.path.join(opt.input_folder, 'high')
val_low_dir = os.path.join(opt.val_folder, 'low')
val_high_dir = os.path.join(opt.val_folder, 'high')

train_files = np.sort([f for f in os.listdir(train_low_dir) if f.lower().endswith(('png','jpg','jpeg'))])
val_files = np.sort([f for f in os.listdir(val_low_dir) if f.lower().endswith(('png','jpg','jpeg'))])

# ----------------------------
# 损失函数定义
# ----------------------------
criterion_sup = torch.nn.MSELoss()
criterion_sasw = SASWLoss()
l_exp, l_TV = L_exp(16, opt.L), L_TV()


# ============================================================
# 训练循环（每张训练图单独训练）
# ============================================================
for img_name in tqdm(train_files):
    input_path = os.path.join(train_low_dir, img_name)
    gt_path = os.path.join(train_high_dir, img_name)
    if not os.path.exists(gt_path):
        tqdm.write(f"⚠️ 跳过 {img_name}（找不到配对的高光图）")
        continue

    # ---- 训练图像 ----
    img_rgb = get_image(input_path)
    gt_rgb = get_image(gt_path)
    img_hsv = rgb2hsv_torch(img_rgb)
    gt_hsv = rgb2hsv_torch(gt_rgb)
    img_v = get_v_component(img_hsv)
    gt_v = get_v_component(gt_hsv)

    # ---- 下采样 ----
    img_v_lr = interpolate_image(img_v, opt.down_size, opt.down_size)
    gt_v_lr = interpolate_image(gt_v, opt.down_size, opt.down_size)
    img_hsv_lr = rgb2hsv_torch(interpolate_image(img_rgb, opt.down_size, opt.down_size))
    gt_hsv_lr = rgb2hsv_torch(interpolate_image(gt_rgb, opt.down_size, opt.down_size))

    # 初始化模型与优化器
    model = INF_MAMBA(in_channels=3, base_channels=opt.base_channels).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    best_val_loss = float('inf')
    save_basename = os.path.splitext(img_name)[0]
    best_ckpt_path = os.path.join(opt.output_folder, f"{save_basename}_best.pth")
    final_ckpt_path = os.path.join(opt.output_folder, f"{save_basename}_final.pth")

    # ----------------------------
    # Epoch 循环
    # ----------------------------
    for epoch in range(opt.epochs):
        model.train()
        optimizer.zero_grad()

        # ---------- 前向 ----------
        #model_input = img_v_lr.repeat(1, 3, 1, 1)
        illu_res_lr = model(img_v_lr)
        illu_lr = illu_res_lr + img_v_lr
        img_v_fixed_lr = img_v_lr / (illu_lr + 1e-4)

        # ---------- 损失计算 ----------
        # 1️⃣ 有监督 V 通道损失
        # MSE损失函数
        loss_sup = criterion_sup(img_v_fixed_lr, gt_v_lr)

        # 2️⃣ SASWLoss (RGB 空间)
        img_hsv_fixed_lr = replace_v_component(img_hsv_lr, img_v_fixed_lr)
        img_rgb_fixed_lr = hsv2rgb_torch(img_hsv_fixed_lr)
        img_rgb_fixed_lr = img_rgb_fixed_lr / torch.max(img_rgb_fixed_lr)
        gt_rgb_lr = hsv2rgb_torch(gt_hsv_lr)
        gt_rgb_lr = gt_rgb_lr / torch.max(gt_rgb_lr)
        #loss_sasw = criterion_sasw(img_rgb_fixed_lr, gt_rgb_lr)
        loss_sasw = criterion_sasw(img_rgb_fixed_lr.detach(), gt_rgb_lr.detach())

        # 3️⃣ 正则项
        loss_tv = l_TV(illu_lr)
        loss_exp = l_exp(illu_lr)
        loss_sparsity = torch.mean(img_v_fixed_lr)

        # 4️⃣ 总损失
        total_loss = (
            opt.lambda_sup * loss_sup +
            opt.lambda_sasw * loss_sasw +
            opt.beta * loss_tv +
            opt.gamma * loss_exp +
            opt.alpha * loss_sparsity
        )

        total_loss.backward()
        optimizer.step()

        # ---------- 验证阶段 ----------
        if (epoch + 1) % opt.val_interval == 0 or epoch == opt.epochs - 1:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vname in val_files:
                    val_low_path = os.path.join(val_low_dir, vname)
                    val_high_path = os.path.join(val_high_dir, vname)
                    if not os.path.exists(val_high_path): continue

                    val_rgb = get_image(val_low_path)
                    val_gt_rgb = get_image(val_high_path)
                    val_hsv = rgb2hsv_torch(val_rgb)
                    val_gt_hsv = rgb2hsv_torch(val_gt_rgb)
                    val_v = get_v_component(val_hsv)
                    val_gt_v = get_v_component(val_gt_hsv)

                    # ---- 下采样 ----
                    img_v_lr = interpolate_image(val_v, opt.down_size, opt.down_size)
                    gt_v_lr = interpolate_image(val_gt_v, opt.down_size, opt.down_size)
                    img_hsv_lr = rgb2hsv_torch(interpolate_image(val_rgb, opt.down_size, opt.down_size))
                    gt_hsv_lr = rgb2hsv_torch(interpolate_image(val_gt_rgb, opt.down_size, opt.down_size))

                    #val_input = val_v.repeat(1, 3, 1, 1).cuda()
                    illu_res_val = model(img_v_lr)
                    illu_val = illu_res_val + img_v_lr
                    img_v_fixed_val = img_v_lr / (illu_val + 1e-4)

                    val_hsv_fixed_lr = replace_v_component(img_hsv_lr, img_v_fixed_val)
                    val_rgb_fixed_lr = hsv2rgb_torch(val_hsv_fixed_lr)
                    val_rgb_fixed_lr = val_rgb_fixed_lr / torch.max(val_rgb_fixed_lr)
                    val_gt_rgb_lr = hsv2rgb_torch(gt_hsv_lr)
                    val_gt_rgb_lr = val_gt_rgb_lr / torch.max(val_gt_rgb_lr)
                    val_loss = criterion_sasw(val_rgb_fixed_lr.detach(), val_gt_rgb_lr.detach())
                    val_losses.append(val_loss.item())

            avg_val_loss = np.mean(val_losses)
            tqdm.write(f"🧪 Epoch {epoch+1}/{opt.epochs} | Eval15 平均 SASWLoss: {avg_val_loss:.6f}")

            # 保存最优模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'img_name': img_name,
                }, best_ckpt_path)
                tqdm.write(f"✅ [BEST] {img_name} | Eval15 SASWLoss: {best_val_loss:.6f}")

        # ---------- 输出训练日志 ----------
        if (epoch + 1) % 50 == 0 or epoch == 0:
            tqdm.write(
                f"{img_name} | Epoch {epoch+1}/{opt.epochs} | "
                f"Sup:{loss_sup.item():.4f} | SASW:{loss_sasw.item():.4f} | "
                f"Total:{total_loss.item():.4f}"
            )

    # ----------------------------
    # 保存最终模型
    # ----------------------------
    torch.save({
        'epoch': opt.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_loss': total_loss.item(),
        'img_name': img_name,
    }, final_ckpt_path)
    tqdm.write(f"💾 Saved final model: {final_ckpt_path}")

    # ----------------------------
    # 推理保存增强结果
    # ----------------------------
    model.eval()
    with torch.no_grad():
        #model_input_full = img_v.repeat(1, 3, 1, 1)
        illu_res_full = model(img_v_lr)
        illu_full = illu_res_full + img_v_lr
        img_v_fixed_lr = img_v_lr / (illu_lr + 1e-4)

        img_v_fixed_full = filter_up(img_v_lr, img_v_fixed_lr, img_v, r=1)
        img_hsv_fixed_full = replace_v_component(img_hsv, img_v_fixed_full)
        img_rgb_fixed_full = hsv2rgb_torch(img_hsv_fixed_full)
        img_rgb_fixed_full = img_rgb_fixed_full / torch.max(img_rgb_fixed_full)

        img_np = (torch.movedim(img_rgb_fixed_full, 1, -1)[0].cpu().detach().numpy() * 255).astype(np.uint8)
        Image.fromarray(img_np).save(os.path.join(opt.output_folder, img_name))

    del model, optimizer, img_v_lr, gt_v_lr
    torch.cuda.empty_cache()
