# ============================================================
# colie_mambalowlight_supervised_rgb_mse_full.py
# 有监督低光增强（一个模型训练整个LOLdataset，使用RGB + MSE损失）
# ============================================================

from utils import *
from loss import *
from siren_mamba import INF_MAMBA
import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F


# ----------------------------
# 参数设置
# ----------------------------
parser = argparse.ArgumentParser(description='CoLIE-Mamba Supervised RGB + MSE (Full Dataset)')
parser.add_argument('--input_folder', type=str, default='input/dataset/LOLdataset/our485',
                    help='训练集主目录 (包含 low, high 子目录)')
parser.add_argument('--val_folder', type=str, default='input/dataset/LOLdataset/eval15',
                    help='验证集主目录 (包含 low, high 子目录)')
parser.add_argument('--output_folder', type=str, default='output/dataset/LOLdataset/train/mamba_rgb_mse_full')
parser.add_argument('--down_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--base_channels', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=3e-4)
parser.add_argument('--val_interval', type=int, default=10, help='每隔多少 epoch 在 eval15 上验证一次')
opt = parser.parse_args()

os.makedirs(opt.output_folder, exist_ok=True)

# ----------------------------
# 数据集路径
# ----------------------------
train_low_dir = os.path.join(opt.input_folder, 'low')
train_high_dir = os.path.join(opt.input_folder, 'high')
val_low_dir = os.path.join(opt.val_folder, 'low')
val_high_dir = os.path.join(opt.val_folder, 'high')

train_files = np.sort([f for f in os.listdir(train_low_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
val_files = np.sort([f for f in os.listdir(val_low_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

# ----------------------------
# 损失函数定义
# ----------------------------
criterion_mse = torch.nn.MSELoss()

# ----------------------------
# 初始化模型与优化器（全局共用）
# ----------------------------
model = INF_MAMBA(in_channels=3, base_channels=opt.base_channels).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

best_val_loss = float('inf')
best_ckpt_path = os.path.join(opt.output_folder, "best_model.pth")
final_ckpt_path = os.path.join(opt.output_folder, "final_model.pth")

# ============================================================
# 训练循环
# ============================================================
for epoch in range(opt.epochs):
    model.train()
    total_train_loss = []

    for img_name in tqdm(train_files, desc=f"Epoch {epoch+1}/{opt.epochs}"):
        input_path = os.path.join(train_low_dir, img_name)
        gt_path = os.path.join(train_high_dir, img_name)
        if not os.path.exists(gt_path):
            continue

        img = get_image(input_path)
        gt_img = get_image(gt_path)

        img_lr = interpolate_image(img, opt.down_size, opt.down_size)
        gt_img_lr = interpolate_image(gt_img, opt.down_size, opt.down_size)

        optimizer.zero_grad()
        output_rgb_lr = model(img_lr)
        enhanced_rgb_lr = torch.clamp(output_rgb_lr + img_lr, 0, 1)

        loss_mse = criterion_mse(enhanced_rgb_lr, gt_img_lr)
        loss_mse.backward()
        optimizer.step()

        total_train_loss.append(loss_mse.item())

    avg_train_loss = np.mean(total_train_loss)
    tqdm.write(f"📘 Epoch {epoch+1}/{opt.epochs} | 训练平均 MSELoss: {avg_train_loss:.6f}")

    # ----------------------------
    # 验证阶段
    # ----------------------------
    if (epoch + 1) % opt.val_interval == 0 or epoch == opt.epochs - 1:
        model.eval()
        val_losses = []
        with torch.no_grad():
            for vname in val_files:
                val_low_path = os.path.join(val_low_dir, vname)
                val_high_path = os.path.join(val_high_dir, vname)
                if not os.path.exists(val_high_path): continue

                val_img = get_image(val_low_path)
                val_gt_img = get_image(val_high_path)
                val_img_lr = interpolate_image(val_img, opt.down_size, opt.down_size)
                val_gt_img_lr = interpolate_image(val_gt_img, opt.down_size, opt.down_size)

                output_val_lr = model(val_img_lr)
                enhanced_val_lr = torch.clamp(output_val_lr + val_img_lr, 0, 1)
                val_loss = criterion_mse(enhanced_val_lr, val_gt_img_lr)
                val_losses.append(val_loss.item())

        avg_val_loss = np.mean(val_losses)
        tqdm.write(f"🧪 Epoch {epoch+1}/{opt.epochs} | Eval15 平均 MSELoss: {avg_val_loss:.6f}")

        # 保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, best_ckpt_path)
            tqdm.write(f"✅ [BEST] 模型更新 | Eval15 MSELoss: {best_val_loss:.6f}")

# ----------------------------
# 保存最终模型
# ----------------------------
torch.save({
    'epoch': opt.epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, final_ckpt_path)
tqdm.write(f"💾 Saved final model: {final_ckpt_path}")

# ============================================================
# 推理阶段（增强所有训练图并保存结果）
# ============================================================
model.eval()
with torch.no_grad():
    for img_name in tqdm(train_files, desc="推理保存增强结果"):
        input_path = os.path.join(train_low_dir, img_name)
        if not os.path.exists(input_path): continue

        img = get_image(input_path)
        output_rgb_full = model(img)
        enhanced_rgb_full = torch.clamp(output_rgb_full + img, 0, 1)

        img_np = (torch.movedim(enhanced_rgb_full, 1, -1)[0].cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img_np).save(os.path.join(opt.output_folder, img_name))

torch.cuda.empty_cache()
tqdm.write("🎉 全部增强结果已保存！")
