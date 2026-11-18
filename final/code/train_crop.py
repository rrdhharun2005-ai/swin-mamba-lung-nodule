"""Crop protocol (as in most LIDC nodule-segmentation papers): train and test on
nodule-centred ROI patches instead of full 512x512 slices.

- Only slices that contain a nodule are used.
- Train: one 128x128 patch per slice around a random nodule, with +-24 px jitter + augmentation.
- Val/test: one 128x128 patch centred on every nodule (connected component), deterministic.
Model: Swin-Mamba U-Net 2.5D (the best full-slice model), same loss/optimiser/schedule.

Usage (from D:/Claude/CV):
    .venv/Scripts/python crop_version/code/train_crop.py --epochs 60
    .venv/Scripts/python crop_version/code/train_crop.py --test-only
Outputs: crop_version/runs/<name>/ best.pt, last.pt, log.csv, test.txt
"""
import argparse, csv, math, random, sys, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

CV = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CV / "code"))
from metrics import SegMeter, dice_bce_loss  # noqa: E402
from model import SwinMambaUNet, count_params  # noqa: E402

DATA = CV / "dataset" / "processed"
RUNS = CV / "crop_version" / "runs"
MEAN, STD = 0.5, 0.25


def load(file, in_ch):
    if in_ch == 3:
        img = cv2.cvtColor(cv2.imread(str(DATA / "images25d" / file), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(str(DATA / "images" / file), cv2.IMREAD_GRAYSCALE)
    mask = (cv2.imread(str(DATA / "masks" / file), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    return img.astype(np.float32) / 255.0, mask


def centroids(mask):
    n, _, _, c = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [(int(round(c[i][1])), int(round(c[i][0]))) for i in range(1, n)]


def cut(img, mask, cy, cx, s):
    H, W = mask.shape
    y0, x0 = int(np.clip(cy - s // 2, 0, H - s)), int(np.clip(cx - s // 2, 0, W - s))
    return img[y0:y0 + s, x0:x0 + s], mask[y0:y0 + s, x0:x0 + s]


class NodulePatches(Dataset):
    def __init__(self, split, size=128, train=False, in_ch=3):
        m = pd.read_csv(DATA / "metadata.csv")
        self.df = m[(m.split == split) & (m.has_nodule == 1)].reset_index(drop=True)
        self.size, self.train, self.in_ch = size, train, in_ch
        self.items = list(range(len(self.df)))
        if not train:  # one fixed patch per nodule component
            self.items = []
            for i, f in enumerate(self.df.file):
                mask = (cv2.imread(str(DATA / "masks" / f), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
                self.items += [(i, cy, cx) for cy, cx in centroids(mask)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        s = self.size
        if self.train:
            img, mask = load(self.df.file[k], self.in_ch)
            cy, cx = random.choice(centroids(mask))
            j = 24
            img, mask = cut(img, mask, cy + random.randint(-j, j), cx + random.randint(-j, j), s)
            if random.random() < 0.5: img, mask = img[:, ::-1], mask[:, ::-1]
            if random.random() < 0.5: img, mask = img[::-1], mask[::-1]
            r = random.randint(0, 3); img, mask = np.rot90(img, r), np.rot90(mask, r)
            if random.random() < 0.5: img = img * random.uniform(0.9, 1.1) + random.uniform(-0.05, 0.05)
            if random.random() < 0.3: img = img + np.random.normal(0, 0.02, img.shape).astype(np.float32)
            if self.in_ch == 3 and random.random() < 0.5: img = img[:, :, ::-1]
            img = np.clip(img, 0, 1)
        else:
            i, cy, cx = self.items[k]
            img, mask = load(self.df.file[i], self.in_ch)
            img, mask = cut(img, mask, cy, cx, s)
        img = torch.from_numpy((np.ascontiguousarray(img) - MEAN) / STD).float()
        img = img.permute(2, 0, 1) if img.ndim == 3 else img.unsqueeze(0)
        return img.contiguous(), torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32)).unsqueeze(0)


class Up2(torch.nn.Module):
    def __init__(self, net):
        super().__init__(); self.net = net

    def forward(self, x):
        F = torch.nn.functional
        y = self.net(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        return F.avg_pool2d(y, 2)


def evaluate(model, dl, dev, amp):
    model.eval(); meter = SegMeter()
    with torch.no_grad():
        for img, mask in dl:
            img, mask = img.to(dev), mask.to(dev)
            with torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
                meter.update(model(img).float(), mask)
    return meter.compute()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="swin_mamba_25d_crop128")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--in-ch", type=int, default=3, choices=[1, 3])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test")
    ap.add_argument("--test-only", action="store_true")
    ap.add_argument("--model", default="swin_mamba", choices=["swin_mamba", "unet", "swin_unet_pt"])
    a = ap.parse_args()
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.bfloat16 if dev.type == "cuda" and torch.cuda.is_bf16_supported() else None
    out = RUNS / a.name; out.mkdir(parents=True, exist_ok=True)
    if a.model == "swin_mamba":
        model = SwinMambaUNet(in_ch=a.in_ch).to(dev)
    else:
        from baselines import MODELS
        model = MODELS[a.model](in_ch=a.in_ch)
        if a.model == "swin_unet_pt":  # window 8 needs >=256 input: same 128 patch, upsampled x2 inside
            model = Up2(model)
        model = model.to(dev)
    mk = lambda ds, bs, sh: DataLoader(ds, bs, shuffle=sh, num_workers=a.workers, pin_memory=True,
                                       drop_last=sh, persistent_workers=a.workers > 0)

    if not a.test_only:
        tr, va = NodulePatches("train", a.size, True, a.in_ch), NodulePatches("val", a.size, False, a.in_ch)
        print(f"params {count_params(model)/1e6:.2f}M | train slices {len(tr)} | val patches {len(va)}", flush=True)
        tdl, vdl = mk(tr, a.batch_size, True), mk(va, a.batch_size, False)
        decay, nod = [], []
        for n, p in model.named_parameters():
            (nod if p.ndim <= 1 or n.endswith(("bias_table", "A_log", "Ds", "dt_bias")) else decay).append(p)
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": nod, "weight_decay": 0}], lr=a.lr)
        spe = a.max_steps or len(tdl); tot, warm = a.epochs * spe, 2 * spe
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                                0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, tot - warm))))
        fields = ["epoch", "train_loss", "dice_global", "dice_pos", "iou_global", "precision", "recall", "time_s"]
        with open(out / "log.csv", "w", newline="") as f: csv.writer(f).writerow(fields)
        best = -1
        for ep in range(a.epochs):
            model.train(); t0, tl, n = time.time(), 0.0, 0
            for step, (img, mask) in enumerate(tdl):
                if a.max_steps and step >= a.max_steps: break
                img, mask = img.to(dev, non_blocking=True), mask.to(dev, non_blocking=True)
                with torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
                    logits = model(img)
                loss = dice_bce_loss(logits.float(), mask)
                opt.zero_grad(set_to_none=True); loss.backward()
                g = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); sch.step()
                if torch.isfinite(loss) and torch.isfinite(g):
                    opt.step(); tl += loss.item(); n += 1
            v = evaluate(model, vdl, dev, amp)
            row = {"epoch": ep, "train_loss": tl / max(n, 1), **{k: v[k] for k in fields[2:7]}, "time_s": time.time() - t0}
            with open(out / "log.csv", "a", newline="") as f: csv.DictWriter(f, fieldnames=fields).writerow(row)
            print(f"epoch {ep}: loss {row['train_loss']:.4f} | val dice {v['dice_global']:.4f} iou {v['iou_global']:.4f} "
                  f"P {v['precision']:.3f} R {v['recall']:.3f} | {row['time_s']:.0f}s", flush=True)
            ck = {"model": model.state_dict(), "epoch": ep, "in_ch": a.in_ch}
            if v["dice_global"] > best:
                best = v["dice_global"]; torch.save(ck, out / "best.pt"); print(f"  new best {best:.4f}", flush=True)
            torch.save(ck, out / "last.pt")

    ck = torch.load(out / "best.pt", map_location=dev); model.load_state_dict(ck["model"])
    te = NodulePatches("test", a.size, False, a.in_ch)
    r = evaluate(model, mk(te, a.batch_size, False), dev, amp)
    txt = (f"crop protocol test | {a.size}x{a.size} nodule-centred patches: {len(te)} | best epoch {ck['epoch']}\n"
           + "\n".join(f"{k}: {r[k]:.4f}" for k in ("dice_global", "dice_pos", "iou_global", "precision", "recall")))
    (out / "test.txt").write_text(txt); print(txt)


if __name__ == "__main__":
    main()
