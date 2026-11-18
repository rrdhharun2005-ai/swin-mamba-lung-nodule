"""Phase 2 pipeline: fine-tuned model, 2D+2.5D ensemble and the retrained refiner, tuned on VALIDATION, tested once.

Candidate stage-1 sources (all probabilities with 4-flip TTA):
  orig      : original 2.5D model (Phase-1 final, step E)
  ft        : fine-tuned 2.5D model
  ft+2d     : mean of fine-tuned 2.5D and 2D Swin-Mamba (2 x 3.08 M)
  ft+orig   : mean of fine-tuned and original 2.5D models (2 x 3.08 M)
Stage-1 cleanup: threshold + minimum blob area (lung mask / slice consistency were never selected in Phase 1).
Stage 2: retrained refiner on 128x128 patches around each kept blob; it may also erase a blob.

Usage (from D:/Claude/CV): .venv/Scripts/python "pseudo final/code/pipeline_v2.py"
"""
import itertools, json, sys, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as P1  # noqa: E402
from pipeline import CACHE, OUT, CV, metrics_from, metrics_masks, crop_probs, log  # noqa: E402
from model import SwinMambaUNet  # noqa: E402

THRS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
AREAS = [0, 10, 20, 30, 50]
dev = torch.device("cuda")


def source(split, name):
    p = CACHE / f"{split}_{name.replace('+', '_')}_mix.npy"
    if name in ("orig", "ft"):
        return np.load(CACHE / f"{split}_{'tta' if name == 'orig' else 'ft_tta'}.npy", mmap_mode="r")
    if p.exists():
        return np.load(p, mmap_mode="r")
    a = np.load(CACHE / f"{split}_ft_tta.npy", mmap_mode="r")
    b = np.load(CACHE / f"{split}_{'2d_tta' if name == 'ft+2d' else 'tta'}.npy", mmap_mode="r")
    M = np.lib.format.open_memmap(p, "w+", np.uint8, a.shape)
    for i in range(0, len(a), 256):
        M[i:i + 256] = ((a[i:i + 256].astype(np.uint16) + b[i:i + 256]) // 2).astype(np.uint8)
    M.flush()
    return M


def comp_table(P, G, thr):
    rows, t8 = [], int(round(thr * 255))
    for s in range(len(P)):
        b = (P[s] >= t8).astype(np.uint8)
        if not b.any():
            continue
        n, lab, st, cen = cv2.connectedComponentsWithStats(b, connectivity=8)
        tp = np.bincount(lab.ravel(), G[s].ravel().astype(np.float64), n)
        rows += [(s, k, st[k, cv2.CC_STAT_AREA], tp[k], cen[k][1], cen[k][0]) for k in range(1, n)]
    return pd.DataFrame(rows, columns=["s", "k", "area", "tp", "cy", "cx"])


def stage2(P, thr, T, probs, n, t2, r, erase):
    t8 = int(round(thr * 255)); masks = [None] * n
    for s, grp in T.groupby("s"):
        _, lab = cv2.connectedComponents((P[s] >= t8).astype(np.uint8), connectivity=8)
        m = np.zeros((512, 512), bool)
        for idx, row in zip(grp.index, grp.itertuples()):
            comp = lab == row.k
            y0, x0, pr = probs[idx]
            reg = comp[y0:y0 + 128, x0:x0 + 128].astype(np.uint8)
            if r:
                reg = cv2.dilate(reg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
            ref = (pr >= t2) & reg.astype(bool)
            if ref.sum() == 0:
                if not erase:
                    m |= comp
                continue
            full = np.zeros((512, 512), bool); full[y0:y0 + 128, x0:x0 + 128] = ref
            m |= full
        masks[s] = m
    return masks


def main():
    t0 = time.time()
    D = {}
    for split in ("val", "test"):
        G = np.load(CACHE / f"{split}_gt.npy", mmap_mode="r")
        D[split] = dict(files=pd.read_csv(CACHE / f"{split}_files.csv").file.tolist(), G=G,
                        gsum=G.reshape(len(G), -1).sum(1).astype(float))
    SRC = ["orig", "ft", "ft+2d", "ft+orig"]
    tables = {}
    for split, src in itertools.product(("val", "test"), SRC):
        Pm = source(split, src)
        for thr in THRS:
            tables[split, src, thr] = comp_table(Pm, D[split]["G"], thr)
        log(f"tables {split} {src} ({time.time() - t0:.0f}s)")

    def keep(T, a): return (T.area >= a).values
    def evalc(split, c):
        T = tables[split, c["src"], c["thr"]]
        return metrics_from(T, keep(T, c["area"]), D[split]["gsum"], len(D[split]["files"]))

    rows, cfgs = [], {}
    def add(name, c, out=None):
        out = out or {s: evalc(s, c) for s in ("val", "test")}
        rows.append(dict(step=name, **{f"val_{k}": v for k, v in out["val"].items()}, **{f"test_{k}": v for k, v in out["test"].items()}))
        cfgs[name] = c
        log(f"{name}: {c} | val {out['val']['dice_global']:.4f} | test {out['test']['dice_global']:.4f} fp {out['test']['fp_slice_rate']:.3f}")

    def best(srcs):
        return max(((evalc("val", c)["dice_global"], c) for c in
                    (dict(src=s, thr=t, area=a) for s, t, a in itertools.product(srcs, THRS, AREAS))), key=lambda z: z[0])[1]

    add("E. Phase-1 final (original model, TTA, cleanup)", dict(src="orig", thr=0.5, area=20))
    add("G. Fine-tuned 2.5D model, re-tuned", best(["ft"]))
    add("H. + 2D/2.5D ensemble options (6.2 M)", best(["ft", "ft+2d", "ft+orig"]))
    base = cfgs[list(cfgs)[-1]]

    ref = SwinMambaUNet(in_ch=3).to(dev)
    ref.load_state_dict(torch.load(OUT.parent / "runs/refiner/best.pt", map_location=dev)["model"]); ref.eval()
    S2 = {}
    for split in ("val", "test"):
        T = tables[split, base["src"], base["thr"]]
        T = T[keep(T, base["area"])].reset_index(drop=True)
        S2[split] = (T, crop_probs(ref, D[split]["files"], T), source(split, base["src"]))
    grid = []
    T, pr, Pm = S2["val"]
    for t2, r, er in itertools.product([0.3, 0.4, 0.5, 0.6, 0.7], [3, 6, 10, 20], [False, True]):
        grid.append((metrics_masks(stage2(Pm, base["thr"], T, pr, len(Pm), t2, r, er), D["val"]["G"])["dice_global"], t2, r, er))
    v, t2, r, er = max(grid)
    log(f"refiner tuned on val: t2={t2} r={r} erase={er} -> val {v:.4f}")
    out = {s: metrics_masks(stage2(S2[s][2], base["thr"], S2[s][0], S2[s][1], len(S2[s][2]), t2, r, er), D[s]["G"]) for s in ("val", "test")}
    add("I. + retrained two-stage refiner", dict(**base, refiner_t=t2, refiner_dilate=r, refiner_erase=er), out)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "final_results_v2.csv", index=False)
    json.dump(cfgs, open(OUT / "pipeline_config_v2.json", "w"), indent=2)
    log("\nTEST SET (105 patients, 2,558 full 512x512 slices)")
    log(df[["step", "test_dice_global", "test_dice_pos", "test_iou_global", "test_precision", "test_recall",
            "test_fp_slice_rate"]].round(4).to_string(index=False))
    log(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
