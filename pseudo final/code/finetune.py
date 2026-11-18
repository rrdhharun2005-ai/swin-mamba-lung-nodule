"""Phase 2: fine-tune the trained 2.5D full-slice Swin-Mamba (runs/swin_mamba_25d/best.pt).

Changes versus the original training (architecture unchanged, 3.08 M parameters):
  * starts from the best checkpoint, lower learning rate (1e-4), short cosine schedule
  * loss = BCE + Dice + 0.5 * focal Tversky (alpha=0.7 on misses, beta=0.3 on false alarms, gamma=0.75)
    -> pushes the model to recover missed nodule pixels; false alarms are handled later by post-processing
  * training crops 256x256 (as before), nodule-centred 90% of the time (was 80%)
Model selection: best validation Dice (full 512x512 val slices), exactly as before.

Usage (from D:/Claude/CV): .venv/Scripts/python "pseudo final/code/finetune.py" --epochs 10
Output: pseudo final/runs/ft_25d/ best.pt, log.csv
"""
import argparse, csv, math, sys, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
CV = ROOT.parent
sys.path.insert(0, str(CV / "code"))
from data import LIDCSlices  # noqa: E402
from metrics import dice_bce_loss  # noqa: E402
from model import SwinMambaUNet  # noqa: E402
from train import evaluate  # noqa: E402


def focal_tversky(logits, target, alpha=0.7, beta=0.3, gamma=0.75, smooth=1.0):
    p = torch.sigmoid(logits)
    tp = (p * target).sum()
    fn = ((1 - p) * target).sum()
    fp = (p * (1 - target)).sum()
    ti = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return (1 - ti) ** gamma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--ft-weight", type=float, default=0.5)
    a = ap.parse_args()
    torch.manual_seed(42)
    dev = torch.device("cuda")
    out = ROOT / "runs" / "ft_25d"
    out.mkdir(parents=True, exist_ok=True)

    train_dl = DataLoader(LIDCSlices("train", crop=a.crop, augment=True, pos_crop_prob=0.9, in_ch=3), a.batch_size,
                          shuffle=True, num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
    val_dl = DataLoader(LIDCSlices("val", in_ch=3), 4, num_workers=4, pin_memory=True, persistent_workers=True)

    model = SwinMambaUNet(in_ch=3).to(dev)
    model.load_state_dict(torch.load(CV / "runs/swin_mamba_25d/best.pt", map_location=dev)["model"])
    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or n_.endswith(("bias_table", "A_log", "Ds", "dt_bias")) else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": no_decay, "weight_decay": 0.0}], lr=a.lr)
    total, warm = a.epochs * len(train_dl), len(train_dl) // 2
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    v0 = evaluate(model, val_dl, dev, torch.bfloat16)
    best = v0["dice_global"]
    print(f"start (original best.pt): val dice {best:.4f} fp {v0['fp_slice_rate']:.3f}", flush=True)
    fields = ["epoch", "train_loss", "loss", "dice_global", "dice_pos", "iou_global", "precision", "recall",
              "fp_slice_rate", "lr", "time_s"]
    with open(out / "log.csv", "w", newline="") as f:
        csv.writer(f).writerow(fields)
        csv.DictWriter(f, fieldnames=fields).writerow({"epoch": -1, "train_loss": 0, **v0, "lr": 0, "time_s": 0})
    torch.save({"model": model.state_dict(), "epoch": -1, "best": best, "in_ch": 3}, out / "best.pt")

    for ep in range(a.epochs):
        model.train()
        t0, run, n = time.time(), 0.0, 0
        for step, (img, mask, _) in enumerate(train_dl):
            img, mask = img.to(dev, non_blocking=True), mask.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(img)
            logits = logits.float()
            loss = dice_bce_loss(logits, mask) + a.ft_weight * focal_tversky(logits, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            g = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            sched.step()
            if not (torch.isfinite(loss) and torch.isfinite(g)):
                continue
            opt.step()
            run += loss.item(); n += 1
            if step % 300 == 0:
                print(f"  ep {ep} step {step}/{len(train_dl)} loss {loss.item():.4f}", flush=True)
        v = evaluate(model, val_dl, dev, torch.bfloat16)
        row = {"epoch": ep, "train_loss": run / max(n, 1), **v, "lr": opt.param_groups[0]["lr"], "time_s": time.time() - t0}
        with open(out / "log.csv", "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(f"epoch {ep}: val dice {v['dice_global']:.4f} (pos {v['dice_pos']:.4f}) recall {v['recall']:.4f} "
              f"fp {v['fp_slice_rate']:.3f} | {row['time_s']:.0f}s", flush=True)
        if v["dice_global"] > best:
            best = v["dice_global"]
            torch.save({"model": model.state_dict(), "epoch": ep, "best": best, "in_ch": 3}, out / "best.pt")
            print(f"  new best {best:.4f}", flush=True)
    print(f"done. best val dice {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
