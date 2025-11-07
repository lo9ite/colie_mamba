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

parser = argparse.ArgumentParser(description='CoLIE with Multi-Direction Mamba')
parser.add_argument('--input_folder', type=str, default='input/dataset/LOLdataset/eval15/low')
parser.add_argument('--output_folder', type=str, default='output/dataset/LOLdataset/eval15/mamba_high_v3_3')
parser.add_argument('--down_size', type=int, default=128, help='train on 128x128 patches')
parser.add_argument('--epochs', type=int, default=800)
parser.add_argument('--base_channels', type=int, default=32)
parser.add_argument('--alpha', type=float, default=85, required=True)
parser.add_argument('--beta', type=float, default=25, required=True)
parser.add_argument('--gamma', type=float, default=43, required=True)
parser.add_argument('--delta', type=float, default=18, required=True)
parser.add_argument('--L', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=3e-4)
opt = parser.parse_args()

os.makedirs(opt.output_folder, exist_ok=True)

for img_name in tqdm(np.sort(os.listdir(opt.input_folder))):
    img_path = os.path.join(opt.input_folder, img_name)
    img_rgb = get_image(img_path)
    img_hsv = rgb2hsv_torch(img_rgb)
    img_v = get_v_component(img_hsv)
    H_org, W_org = img_v.shape[2:]

    # 下采样到128×128
    img_v_lr = interpolate_image(img_v, opt.down_size, opt.down_size)

    model = INF_MAMBA(in_channels=3, base_channels=opt.base_channels).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    l_exp, l_TV = L_exp(16, opt.L), L_TV()

    model.train()
    for epoch in range(opt.epochs):
        optimizer.zero_grad()
        illu_res_lr = model(img_v_lr)
        illu_lr = illu_res_lr + img_v_lr
        img_v_fixed_lr = img_v_lr / (illu_lr + 1e-4)

        loss_fidelity = F.mse_loss(illu_lr, img_v_lr)
        loss_tv = l_TV(illu_lr)
        loss_exp = l_exp(illu_lr)
        loss_sparsity = torch.mean(img_v_fixed_lr)
        criterion = SASWLoss()

        total_loss = (
            opt.delta * loss_fidelity +
            opt.beta * loss_tv +
            opt.gamma * loss_exp +
            opt.alpha * loss_sparsity
        )
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            tqdm.write(f"{img_name} | Epoch {epoch+1}/{opt.epochs} | Loss: {total_loss.item():.4f}")

    img_v_fixed = filter_up(img_v_lr, img_v_fixed_lr, img_v, r=1)
    img_hsv_fixed = replace_v_component(img_hsv, img_v_fixed)
    img_rgb_fixed = hsv2rgb_torch(img_hsv_fixed)
    img_rgb_fixed = img_rgb_fixed / torch.max(img_rgb_fixed)

    img_np = (torch.movedim(img_rgb_fixed, 1, -1)[0].cpu().detach().numpy() * 255).astype(np.uint8)
    Image.fromarray(img_np).save(os.path.join(opt.output_folder, img_name))
