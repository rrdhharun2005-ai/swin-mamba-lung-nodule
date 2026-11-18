"""Step 1 of the final pipeline: run the trained models once and cache their outputs (no training).

For every val/test slice it stores
  * stage-1 probability of the 2.5D full-slice Swin-Mamba (runs/swin_mamba_25d/best.pt)
      - raw  : one forward pass
      - tta  : mean over 4 flips (none, horizontal, vertical, both)
  * neighbour masks for the slice-consistency check: the 2D Swin-Mamba (runs/swin_mamba/best.pt)
    applied to the two neighbour slices that are already inside each 2.5D image (channels z-k and z+k).
  * ground-truth mask
Probabilities are stored as uint8 (p*255) memmaps in pseudo final/cache/.

Usage (from D:/Claude/CV):  .venv/Scripts/python "pseudo final/code/predict.py"
"""
import sys, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

CV = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CV / "code"))
from model import SwinMambaUNet  # noqa: E402

DATA = CV / "dataset" / "processed"
CACHE = Path(__file__).resolve().parents[1] / "cache"
MEAN, STD, BS = 0.5, 0.25, 4
dev = torch.device("cuda")


def load_model(path, in_ch):
    m = SwinMambaUNet(in_ch=in_ch).to(dev)
    m.load_state_dict(torch.load(path, map_location=dev)["model"])
    return m.eval()


@torch.no_grad()
def prob(model, x, tta):
    flips = [[], [3], [2], [2, 3]] if tta else [[]]
    out = 0
    for f in flips:
        xi = x.flip(f) if f else x
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.sigmoid(model(xi).float())
        out = out + (p.flip(f) if f else p)
    return (out / len(flips))[:, 0]


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    s1 = load_model(CV / "runs/swin_mamba_25d/best.pt", 3)
    nb = load_model(CV / "runs/swin_mamba/best.pt", 1)
    meta = pd.read_csv(DATA / "metadata.csv")
    for split in ("val", "test"):
        df = meta[meta.split == split].reset_index(drop=True)
        df.to_csv(CACHE / f"{split}_files.csv", index=False)
        N = len(df)
        mm = {k: np.lib.format.open_memmap(CACHE / f"{split}_{k}.npy", "w+", np.uint8, (N, 512, 512))
              for k in ("raw", "tta", "gt", "nb_above", "nb_below")}
        t0 = time.time()
        for i0 in range(0, N, BS):
            files = df.file[i0:i0 + BS]
            imgs = np.stack([cv2.cvtColor(cv2.imread(str(DATA / "images25d" / f)), cv2.COLOR_BGR2RGB) for f in files])
            x = torch.from_numpy((imgs.astype(np.float32) / 255 - MEAN) / STD).permute(0, 3, 1, 2).to(dev)
            sl = slice(i0, i0 + len(files))
            mm["raw"][sl] = (prob(s1, x, False) * 255).round().byte().cpu().numpy()
            mm["tta"][sl] = (prob(s1, x, True) * 255).round().byte().cpu().numpy()
            mm["nb_above"][sl] = (prob(nb, x[:, 0:1], False) > 0.5).byte().cpu().numpy()
            mm["nb_below"][sl] = (prob(nb, x[:, 2:3], False) > 0.5).byte().cpu().numpy()
            mm["gt"][sl] = np.stack([cv2.imread(str(DATA / "masks" / f), 0) > 127 for f in files])
            if (i0 // BS) % 100 == 0:
                print(f"{split} {i0 + len(files)}/{N}  {time.time() - t0:.0f}s", flush=True)
        for a in mm.values():
            a.flush()
        print(f"{split} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
