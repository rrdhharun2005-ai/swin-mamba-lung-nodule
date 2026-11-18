"""Evaluate a trained checkpoint on the test split (or val), report metrics, model size and speed.

Usage (from the CV folder):
    .venv/Scripts/python code/evaluate.py --ckpt runs/swin_mamba/best.pt
    .venv/Scripts/python code/evaluate.py --ckpt runs/swin_mamba/best.pt --vis 20   # save overlays
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import MEAN, STD, LIDCSlices
from metrics import SegMeter
from model import SwinMambaUNet, count_params


def overlay(img, gt, pred):
    """grey CT + green ground truth contour + red prediction contour"""
    base = cv2.cvtColor(((img * STD + MEAN).clip(0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for m, color in ((gt, (0, 255, 0)), (pred, (0, 0, 255))):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(base, cs, -1, color, 1)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--vis", type=int, default=0, help="save N overlay images of nodule slices")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    ck = torch.load(a.ckpt, map_location=device)
    in_ch = ck.get("in_ch", 1)
    model = SwinMambaUNet(in_ch=in_ch).to(device).eval()
    model.load_state_dict(ck["model"])

    ds = LIDCSlices(a.split, in_ch=in_ch)
    dl = DataLoader(ds, a.batch_size, num_workers=4)
    meter = SegMeter()
    vis_dir = Path(a.ckpt).parent / f"vis_{a.split}"
    saved = 0
    with torch.no_grad():
        for img, mask, has in dl:
            img, mask = img.to(device), mask.to(device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(img)
            meter.update(logits, mask)
            if saved < a.vis:
                pred = (torch.sigmoid(logits.float()) > 0.5).cpu().numpy()
                for j in range(img.size(0)):
                    if has[j] and saved < a.vis:
                        vis_dir.mkdir(exist_ok=True)
                        cv2.imwrite(str(vis_dir / f"{saved:03d}.png"),
                                    overlay(img[j, in_ch // 2].cpu().numpy(), mask[j, 0].cpu().numpy(), pred[j, 0]))
                        saved += 1

    print(f"{a.split} set: {len(ds)} slices")
    for k, v in meter.compute().items():
        print(f"  {k:14s} {v:.4f}")

    # model size and speed (single 512x512 slice)
    x = torch.randn(1, in_ch, 512, 512, device=device)
    with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        for _ in range(5):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        for _ in range(20):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    print(f"  params         {count_params(model) / 1e6:.2f} M")
    print(f"  speed          {20 / (time.time() - t):.1f} slices/s (512x512, {device})")
    if saved:
        print(f"  overlays saved to {vis_dir} (green = ground truth, red = prediction)")


if __name__ == "__main__":
    main()
