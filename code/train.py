"""Train the Lightweight Swin-Mamba U-Net on LIDC-IDRI 2D slices.

Usage (from the CV folder):
    .venv/Scripts/python code/train.py --epochs 60 --batch-size 16 --name swin_mamba
    .venv/Scripts/python code/train.py --resume runs/swin_mamba/last.pt   # continue a run
Outputs go to runs/<name>/ : best.pt (by val Dice), last.pt, log.csv
"""
import argparse
import csv
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import LIDCSlices
from metrics import SegMeter, dice_bce_loss
from model import SwinMambaUNet, count_params

RUNS = Path(__file__).resolve().parent.parent / "runs"


def evaluate(model, loader, device, amp_dtype):
    model.eval()
    meter, loss_sum, n = SegMeter(), 0.0, 0
    with torch.no_grad():
        for img, mask, _ in loader:
            img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(img)
            loss_sum += dice_bce_loss(logits.float(), mask).item() * img.size(0)
            n += img.size(0)
            meter.update(logits, mask)
    return {"loss": loss_sum / max(n, 1), **meter.compute()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="swin_mamba")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256, help="training crop size (multiple of 32)")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=2, help="warm-up epochs")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=0, help="limit steps per epoch (smoke test)")
    ap.add_argument("--resume", default="")
    ap.add_argument("--in-ch", type=int, default=1, choices=[1, 3], help="1 = single slice, 3 = 2.5D")
    a = ap.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    out = RUNS / a.name
    out.mkdir(parents=True, exist_ok=True)

    train_ds = LIDCSlices("train", crop=a.crop, augment=True, in_ch=a.in_ch)
    val_ds = LIDCSlices("val", in_ch=a.in_ch)
    train_dl = DataLoader(train_ds, a.batch_size, shuffle=True, num_workers=a.workers,
                          pin_memory=True, drop_last=True, persistent_workers=a.workers > 0)
    val_dl = DataLoader(val_ds, max(1, a.batch_size // 4), num_workers=a.workers,
                        pin_memory=True, persistent_workers=a.workers > 0)

    model = SwinMambaUNet(in_ch=a.in_ch).to(device)
    print(f"params: {count_params(model) / 1e6:.2f} M | train {len(train_ds)} | val {len(val_ds)} | {device}")

    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or n_.endswith(("bias_table", "A_log", "Ds", "dt_bias")) else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=a.lr)
    steps_per_epoch = a.max_steps or len(train_dl)
    total, warm = a.epochs * steps_per_epoch, a.warmup * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    start, best = 0, -1.0
    if a.resume:
        ck = torch.load(a.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed from epoch {ck['epoch']} (best val dice {best:.4f})")

    log_path = out / "log.csv"
    fields = ["epoch", "train_loss", "loss", "dice_global", "dice_pos", "iou_global", "precision",
              "recall", "fp_slice_rate", "lr", "time_s"]
    if not log_path.exists() or not a.resume:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(fields)

    for epoch in range(start, a.epochs):
        model.train()
        t0, run_loss, n_ok, skipped = time.time(), 0.0, 0, 0
        for step, (img, mask, _) in enumerate(train_dl):
            if a.max_steps and step >= a.max_steps:
                break
            img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(img)
            loss = dice_bce_loss(logits.float(), mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            sched.step()
            if not (torch.isfinite(loss) and torch.isfinite(gnorm)):  # skip bad batch, keep weights intact
                skipped += 1
                if skipped > 50:
                    raise RuntimeError("too many non-finite steps; training diverged")
                continue
            opt.step()
            run_loss += loss.item()
            n_ok += 1
            if step % 100 == 0:
                print(f"  ep {epoch} step {step}/{steps_per_epoch} loss {loss.item():.4f}", flush=True)

        val = evaluate(model, val_dl, device, amp_dtype)
        if skipped:
            print(f"  warning: skipped {skipped} non-finite steps this epoch")
        row = {"epoch": epoch, "train_loss": run_loss / max(n_ok, 1), **val,
               "lr": opt.param_groups[0]["lr"], "time_s": time.time() - t0}
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(f"epoch {epoch}: train_loss {row['train_loss']:.4f} | val dice {val['dice_global']:.4f} "
              f"(pos-slice {val['dice_pos']:.4f}) iou {val['iou_global']:.4f} recall {val['recall']:.4f} "
              f"fp-slices {val['fp_slice_rate']:.3f} | {row['time_s']:.0f}s", flush=True)

        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
              "epoch": epoch, "best": max(best, val["dice_global"]), "in_ch": a.in_ch}
        if val["dice_global"] > best:
            best = val["dice_global"]
            ck["best"] = best
            torch.save(ck, out / "best.pt")
            print(f"  new best val dice {best:.4f} -> saved best.pt")
        torch.save(ck, out / "last.pt")

    print(f"done. best val dice {best:.4f}")


if __name__ == "__main__":
    main()
