"""Phase 2 predictions (no training): run the fine-tuned 2.5D model and the 2D model once.

Writes to pseudo final/cache/:
  {val,test}_ft_tta.npy : fine-tuned 2.5D Swin-Mamba, 4-flip TTA probability (uint8)
  {val,test}_2d_tta.npy : 2D Swin-Mamba on the centre slice, 4-flip TTA probability (uint8) -> for a 2D+2.5D ensemble
  {train,val}_cands.csv : stage-1 candidate blobs (fine-tuned model, single pass, p>=0.5) used to train the refiner

Usage (from D:/Claude/CV): .venv/Scripts/python "pseudo final/code/predict_v2.py"
"""
import sys, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
CV = ROOT.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import DATA, CACHE, BS, MEAN, STD, load_model, prob  # noqa: E402


def batches(df):
    for i0 in range(0, len(df), BS):
        files = df.file[i0:i0 + BS].tolist()
        imgs = np.stack([cv2.cvtColor(cv2.imread(str(DATA / "images25d" / f)), cv2.COLOR_BGR2RGB) for f in files])
        x = torch.from_numpy((imgs.astype(np.float32) / 255 - MEAN) / STD).permute(0, 3, 1, 2).cuda()
        yield i0, files, x


def candidates(files, P):
    rows = []
    for f, p in zip(files, P):
        n, _, st, cen = cv2.connectedComponentsWithStats((p >= 0.5).astype(np.uint8), connectivity=8)
        rows += [(f, int(round(cen[k][1])), int(round(cen[k][0])), int(st[k, cv2.CC_STAT_AREA])) for k in range(1, n)]
    return rows


def main():
    ft = load_model(ROOT / "runs/ft_25d/best.pt", 3)
    m2d = load_model(CV / "runs/swin_mamba/best.pt", 1)
    meta = pd.read_csv(DATA / "metadata.csv")
    t0 = time.time()
    for split in ("val", "test"):
        df = pd.read_csv(CACHE / f"{split}_files.csv")
        A = np.lib.format.open_memmap(CACHE / f"{split}_ft_tta.npy", "w+", np.uint8, (len(df), 512, 512))
        B = np.lib.format.open_memmap(CACHE / f"{split}_2d_tta.npy", "w+", np.uint8, (len(df), 512, 512))
        cands = []
        for i0, files, x in batches(df):
            p = prob(ft, x, True)
            A[i0:i0 + len(files)] = (p * 255).round().byte().cpu().numpy()
            B[i0:i0 + len(files)] = (prob(m2d, x[:, 1:2], True) * 255).round().byte().cpu().numpy()
            if split == "val":
                cands += candidates(files, prob(ft, x, False).cpu().numpy())
        A.flush(); B.flush()
        if split == "val":
            pd.DataFrame(cands, columns=["file", "cy", "cx", "area"]).to_csv(CACHE / "val_cands.csv", index=False)
        print(f"{split} done {time.time() - t0:.0f}s", flush=True)
    df = meta[meta.split == "train"].reset_index(drop=True)
    cands = []
    for i0, files, x in batches(df):
        cands += candidates(files, prob(ft, x, False).cpu().numpy())
        if (i0 // BS) % 500 == 0:
            print(f"train {i0}/{len(df)} {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(cands, columns=["file", "cy", "cx", "area"]).to_csv(CACHE / "train_cands.csv", index=False)
    print(f"train candidates: {len(cands)} | done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
