"""Phase 2: retrain the stage-2 refiner on what stage 1 actually produces.

The first hybrid failed because the crop model had only seen 128x128 patches centred exactly on real
nodules. Here the training patches are
  * centred on the fine-tuned stage-1 model's candidate blobs on TRAIN slices (true and false ones;
    a false candidate has an empty target, so the refiner learns to erase it), plus
  * centred on the real nodules (as before), so it keeps its outlining skill.
Starts from the crop model (crop_version/runs/swin_mamba_25d_crop128/best.pt), same architecture (3.08 M).
Selection: best Dice on patches around the VALIDATION candidates.

Usage (from D:/Claude/CV): .venv/Scripts/python "pseudo final/code/train_refiner.py" --epochs 15
Output: pseudo final/runs/refiner/ best.pt, log.csv
"""
import argparse, csv, math, random, sys, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
CV = ROOT.parent
sys.path.insert(0, str(CV / "code"))
sys.path.insert(0, str(CV / "crop_version" / "code"))
from metrics import SegMeter, dice_bce_loss  # noqa: E402
from model import SwinMambaUNet  # noqa: E402
from train_crop import centroids, load  # noqa: E402

CACHE = ROOT / "cache"
DATA = CV / "dataset" / "processed"


class Patches(Dataset):
    def __init__(self, split, train):
        c = pd.read_csv(CACHE / f"{split}_cands.csv")
        items = list(zip(c.file, c.cy, c.cx))
        if train:
            m = pd.read_csv(DATA / "metadata.csv")
            for f in m[(m.split == split) & (m.has_nodule == 1)].file:
                mask = (cv2.imread(str(DATA / "masks" / f), 0) > 127).astype(np.uint8)
                items += [(f, cy, cx) for cy, cx in centroids(mask)]
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, cy, cx = self.items[i]
        img, mask = load(f, 3)
        if self.train:
            cy += random.randint(-8, 8); cx += random.randint(-8, 8)
        y0, x0 = int(np.clip(cy - 64, 0, 384)), int(np.clip(cx - 64, 0, 384))
        img, mask = img[y0:y0 + 128, x0:x0 + 128], mask[y0:y0 + 128, x0:x0 + 128]
        if self.train:
            if random.random() < 0.5: img, mask = img[:, ::-1], mask[:, ::-1]
            if random.random() < 0.5: img, mask = img[::-1], mask[::-1]
            r = random.randint(0, 3); img, mask = np.rot90(img, r), np.rot90(mask, r)
            if random.random() < 0.5: img = np.clip(img * random.uniform(0.9, 1.1) + random.uniform(-0.05, 0.05), 0, 1)
            if random.random() < 0.5: img = img[:, :, ::-1]
        x = torch.from_numpy((np.ascontiguousarray(img) - 0.5) / 0.25).permute(2, 0, 1).float()
        return x, torch.from_numpy(np.ascontiguousarray(mask)).float()[None]


@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval(); m = SegMeter()
    for x, y in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m.update(model(x.to(dev)).float(), y.to(dev))
    return m.compute()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()
    torch.manual_seed(42); random.seed(42)
    dev = torch.device("cuda")
    out = ROOT / "runs" / "refiner"; out.mkdir(parents=True, exist_ok=True)
    tr, va = Patches("train", True), Patches("val", False)
    print(f"train patches {len(tr)} | val candidate patches {len(va)}", flush=True)
    tdl = DataLoader(tr, 32, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    vdl = DataLoader(va, 32, num_workers=4, persistent_workers=True)
    model = SwinMambaUNet(in_ch=3).to(dev)
    model.load_state_dict(torch.load(CV / "crop_version/runs/swin_mamba_25d_crop128/best.pt", map_location=dev)["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    total = a.epochs * len(tdl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / total)))
    v = evaluate(model, vdl, dev); best = v["dice_global"]
    print(f"start (original crop model) on val candidates: dice {best:.4f}", flush=True)
    torch.save({"model": model.state_dict(), "epoch": -1, "best": best}, out / "best.pt")
    fields = ["epoch", "train_loss", *v.keys(), "time_s"]
    with open(out / "log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerow({"epoch": -1, "train_loss": 0, **v, "time_s": 0})
    for ep in range(a.epochs):
        model.train(); t0, run, n = time.time(), 0.0, 0
        for x, y in tdl:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(x)
            loss = dice_bce_loss(lg.float(), y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            run += loss.item(); n += 1
        v = evaluate(model, vdl, dev)
        with open(out / "log.csv", "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow({"epoch": ep, "train_loss": run / n, **v, "time_s": time.time() - t0})
        print(f"epoch {ep}: val-candidate dice {v['dice_global']:.4f} precision {v['precision']:.4f} recall {v['recall']:.4f} "
              f"| {time.time() - t0:.0f}s", flush=True)
        if v["dice_global"] > best:
            best = v["dice_global"]
            torch.save({"model": model.state_dict(), "epoch": ep, "best": best}, out / "best.pt")
            print(f"  new best {best:.4f}", flush=True)
    print(f"done. best {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
