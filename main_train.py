import argparse
import os
import random
import time

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SegDataset


# -----------------------
# Utils
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_loss(pred, target, eps=1e-6):
    # pred: logits (B,1,H,W); target: 0/1 (B,1,H,W)
    pred = torch.sigmoid(pred)
    num = 2.0 * (pred * target).sum(dim=(2, 3)) + eps
    den = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    return 1.0 - (num / den).mean()


def iou_score(pred, target, thresh=0.5, eps=1e-6):
    # pred: logits
    pred_bin = (torch.sigmoid(pred) > thresh).float()
    inter = (pred_bin * target).sum(dim=(2, 3))
    union = (pred_bin + target - pred_bin * target).sum(dim=(2, 3)) + eps
    return ((inter + eps) / union).mean().item()


# -----------------------
# Model: EVA02 backbone + simple upsampling decoder
# -----------------------
class EvaSeg(nn.Module):
    def __init__(self, backbone_name="eva02_large_patch14_448", num_classes=1, image_size=448):
        super().__init__()
        self.image_size = image_size
        self.patch_size = 14
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool="")
        with torch.no_grad():
            _feat = self.backbone.forward_features(torch.zeros(1, 3, image_size, image_size))
        embed_dim = _feat.shape[-1]

        dec_channels = [embed_dim, 512, 256, 128, 64, 32]
        blocks = []
        in_ch = dec_channels[0]
        for out_ch in dec_channels[1:]:
            blocks += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        self.decoder = nn.Sequential(*blocks)
        self.final_conv = nn.Conv2d(in_ch, num_classes, 1)

    def tokens_to_map(self, x):
        B, L, C = x.shape
        h = w = self.image_size // self.patch_size
        spatial = h * w
        if L == spatial + 1:
            x = x[:, 1:, :]
        elif L == spatial + 2:
            x = x[:, 2:, :]
        elif L != spatial:
            x = x[:, -spatial:, :]
        x = x.transpose(1, 2).contiguous().view(B, C, h, w)
        return x

    def forward(self, x):
        B, _, H, W = x.shape
        feats = self.backbone.forward_features(x)  # (B, L, C) with cls token
        fmap = self.tokens_to_map(feats)  # (B, C, H/14, W/14)
        y = self.decoder(fmap)
        y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        y = self.final_conv(y)  # (B,1,H,W) logits
        return y


# -----------------------
# Train / Val loops
# -----------------------
def train_one_epoch(model, loader, optimizer, scaler, device, bce_weight=0.5):
    model.train()
    running = {"loss": 0.0, "dice": 0.0, "iou": 0.0}
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True):
            logits = model(x)
            bce = F.binary_cross_entropy_with_logits(logits, y)
            dsc = dice_loss(logits, y)
            loss = bce_weight * bce + (1.0 - bce_weight) * dsc
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running["loss"] += loss.item() * x.size(0)
        running["dice"] += (1.0 - dsc).item() * x.size(0)
        running["iou"] += iou_score(logits.detach(), y) * x.size(0)

    n = len(loader.dataset)
    return {k: v / n for k, v in running.items()}


@torch.no_grad()
def validate(model, loader, device, bce_weight=0.5):
    model.eval()
    running = {"loss": 0.0, "dice": 0.0, "iou": 0.0}
    for x, y in tqdm(loader, desc="valid", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        bce = F.binary_cross_entropy_with_logits(logits, y)
        dsc = dice_loss(logits, y)
        loss = bce_weight * bce + (1.0 - bce_weight) * dsc

        running["loss"] += loss.item() * x.size(0)
        running["dice"] += (1.0 - dsc).item() * x.size(0)
        running["iou"] += iou_score(logits, y) * x.size(0)
    n = len(loader.dataset)
    return {k: v / n for k, v in running.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--backbone", type=str, default="eva02_large_patch14_448")
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--bce_weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--outdir", type=str, default="runs")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, {torch.cuda.get_device_name(0)}")

    # Datasets
    train_ds = SegDataset(args.data_root, "train", image_size=args.image_size)
    val_ds = SegDataset(args.data_root, "val", image_size=args.image_size)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # Model
    model = EvaSeg(backbone_name=args.backbone, num_classes=1, image_size=args.image_size)
    model.to(device)

    # Optim / AMP
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    best_iou = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_dl, optimizer, scaler, device, args.bce_weight)
        va = validate(model, val_dl, device, args.bce_weight)
        dt = time.time() - t0
        print(f"[{epoch:03d}/{args.epochs}] "
              f"train: loss={tr['loss']:.4f} dice={tr['dice']:.4f} iou={tr['iou']:.4f} | "
              f"val: loss={va['loss']:.4f} dice={va['dice']:.4f} iou={va['iou']:.4f} | "
              f"{dt:.1f}s")

        # save last + best
        torch.save({"model": model.state_dict(), "epoch": epoch}, os.path.join(args.outdir, "last.pt"))
        if va["iou"] > best_iou:
            best_iou = va["iou"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_iou": best_iou},
                       os.path.join(args.outdir, "best.pt"))
            print(f"✓ new best IoU: {best_iou:.4f}")

    with torch.no_grad():
        x, y = val_ds[0]
        x1 = x.unsqueeze(0).to(device)
        logits = model(x1)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred = (prob > 0.5).astype(np.uint8) * 255

        os.makedirs(args.outdir, exist_ok=True)
        vis_img = (x.permute(1, 2, 0).cpu().numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        vis_img = (np.clip(vis_img, 0, 1) * 255).astype(np.uint8)[:, :, ::-1]  # в BGR для cv2.imwrite
        cv2.imwrite(os.path.join(args.outdir, "val0_image.jpg"), vis_img)
        cv2.imwrite(os.path.join(args.outdir, "val0_pred.png"), pred)
        cv2.imwrite(os.path.join(args.outdir, "val0_mask.png"), (y[0].cpu().numpy() * 255).astype(np.uint8))
        print("Saved samples: val0_image.jpg / val0_pred.png / val0_mask.png")


if __name__ == "__main__":
    main()
