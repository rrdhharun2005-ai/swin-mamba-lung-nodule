"""Step 2 of the final pipeline: post-processing + two-stage refinement, tuned on VALIDATION, tested once.

Inputs: cache/ written by predict.py (no training anywhere).
Stage 1 cleanup, applied per connected component of (prob >= threshold):
  * min_area  : drop blobs smaller than A pixels
  * lung      : keep a blob only if >= 50% of it lies inside the lung field (dilated by d px)
  * consistency: keep a blob only if the 2D model also finds an overlapping blob on the slice
                 above or below (neighbour channels of the 2.5D image), for blobs smaller than area cap
Stage 2 (two-stage hybrid): for every kept blob, cut a 128x128 patch around it, segment it with the
crop-trained 2.5D Swin-Mamba (4-flip TTA), keep the refined pixels inside the blob dilated by r px,
and paste them back into the 512x512 mask. Dice is always computed on the full 512x512 slices.

Usage: .venv/Scripts/python "pseudo final/code/pipeline.py"
"""
import itertools, json, sys, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
CV = ROOT.parent
sys.path.insert(0, str(CV / "code"))
from model import SwinMambaUNet  # noqa: E402

DATA, CACHE, OUT = CV / "dataset" / "processed", ROOT / "cache", ROOT / "results"
THRS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
AREAS = [0, 5, 10, 20, 30, 50]
LUNG_D = [None, 0, 5, 10, 20]
CONS = [None, 50, 100, 10 ** 9]        # area cap below which the consistency rule applies (None = off)
MIN_AREA_FP = 10                      # same false-alarm definition as the original evaluation
dev = torch.device("cuda")


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ lung field
def lung_mask(img):
    """img: uint8 lung-window slice. Returns uint8 0/1 lung field (holes filled), or all ones if not found."""
    dark = (cv2.GaussianBlur(img, (5, 5), 0) < 100).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    comps = [(st[i, cv2.CC_STAT_AREA], i) for i in range(1, n) if i not in border and st[i, cv2.CC_STAT_AREA] > 1500]
    if not comps:
        return np.ones_like(img)
    m = np.isin(lab, [i for _, i in sorted(comps, reverse=True)[:2]]).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    ff = m.copy()
    cv2.floodFill(ff, np.zeros((514, 514), np.uint8), (0, 0), 1)
    return m | (1 - ff)                 # fill internal holes (vessels, nodules)


def lung_cache(split, files):
    p = CACHE / f"{split}_lung.npy"
    if p.exists():
        return np.load(p, mmap_mode="r")
    L = np.lib.format.open_memmap(p, "w+", np.uint8, (len(files), 512, 512))
    for i, f in enumerate(files):
        L[i] = lung_mask(cv2.imread(str(DATA / "images" / f), 0))
    L.flush()
    return L


# ------------------------------------------------------------------ component tables
KER = {d: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1)) for d in (3, 5, 10, 20)}


def comp_table(P, G, L, NA, NB, thr):
    """One row per connected component of P>=thr with everything needed to evaluate any filter."""
    rows, t8 = [], int(round(thr * 255))
    for s in range(len(P)):
        b = (P[s] >= t8).astype(np.uint8)
        if not b.any():
            continue
        n, lab, st, cen = cv2.connectedComponentsWithStats(b, connectivity=8)
        f = lab.ravel()
        g = G[s].ravel().astype(np.float64)
        tp = np.bincount(f, g, n)
        lung = {0: np.bincount(f, L[s].ravel().astype(np.float64), n)}
        for d in (5, 10, 20):
            lung[d] = np.bincount(f, cv2.dilate(L[s], KER[d]).ravel().astype(np.float64), n)
        nbm = cv2.dilate(np.maximum(NA[s], NB[s]), KER[3]).ravel().astype(np.float64)
        nbo = np.bincount(f, nbm, n)
        for k in range(1, n):
            a = st[k, cv2.CC_STAT_AREA]
            rows.append((s, k, a, tp[k], *(lung[d][k] / a for d in (0, 5, 10, 20)), nbo[k] > 0, cen[k][1], cen[k][0]))
    cols = ["s", "k", "area", "tp", "l0", "l5", "l10", "l20", "nb", "cy", "cx"]
    return pd.DataFrame(rows, columns=cols)


def keep_mask(T, area, lung, cons):
    k = T.area >= area
    if lung is not None:
        k &= T[f"l{lung}"] >= 0.5
    if cons is not None:
        k &= T.nb | (T.area >= cons)
    return k.values


def metrics_from(T, keep, gsum, n):
    """Exact full-slice metrics from per-component counts (components are disjoint)."""
    tps = np.bincount(T.s.values[keep], T.tp.values[keep], n)
    ps = np.bincount(T.s.values[keep], T.area.values[keep], n)
    TP, P, G = tps.sum(), ps.sum(), gsum.sum()
    pos = gsum > 0
    return dict(dice_global=2 * TP / (P + G), dice_pos=float(np.mean(2 * tps[pos] / (ps[pos] + gsum[pos]))),
                iou_global=TP / (P + G - TP), precision=TP / max(P, 1), recall=TP / G,
                fp_slice_rate=float(np.mean(ps[~pos] >= MIN_AREA_FP)))


def metrics_masks(pred_list, G):
    """Metrics from explicit predicted masks (list aligned with slices; None = empty)."""
    tp = np.zeros(len(G)); ps = np.zeros(len(G)); gs = np.array([g.sum() for g in G], float)
    for s, m in enumerate(pred_list):
        if m is not None:
            ps[s] = m.sum(); tp[s] = (m & G[s].astype(bool)).sum()
    pos = gs > 0
    TP, P, Gt = tp.sum(), ps.sum(), gs.sum()
    return dict(dice_global=2 * TP / (P + Gt), dice_pos=float(np.mean(2 * tp[pos] / (ps[pos] + gs[pos]))),
                iou_global=TP / (P + Gt - TP), precision=TP / max(P, 1), recall=TP / Gt,
                fp_slice_rate=float(np.mean(ps[~pos] >= MIN_AREA_FP)))


# ------------------------------------------------------------------ stage 2
def load_crop_model():
    m = SwinMambaUNet(in_ch=3).to(dev)
    m.load_state_dict(torch.load(CV / "crop_version/runs/swin_mamba_25d_crop128/best.pt", map_location=dev)["model"])
    return m.eval()


@torch.no_grad()
def crop_probs(model, files, T):
    """4-flip TTA probability (128x128) for every kept component; returns list aligned with T rows."""
    out, batch, meta = [None] * len(T), [], []
    cache = {}
    def flush():
        if not batch:
            return
        x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).to(dev)
        p = 0
        for f in ([], [3], [2], [2, 3]):
            xi = x.flip(f) if f else x
            with torch.autocast("cuda", dtype=torch.bfloat16):
                q = torch.sigmoid(model(xi).float())
            p = p + (q.flip(f) if f else q)
        p = (p / 4)[:, 0].cpu().numpy()
        for (r, y0, x0), pr in zip(meta, p):
            out[r] = (y0, x0, pr)
        batch.clear(); meta.clear()
    for r, row in enumerate(T.itertuples()):
        f = files[row.s]
        if f not in cache:
            cache.clear()
            img = cv2.cvtColor(cv2.imread(str(DATA / "images25d" / f)), cv2.COLOR_BGR2RGB)
            cache[f] = (img.astype(np.float32) / 255 - 0.5) / 0.25
        img = cache[f]
        y0 = int(np.clip(round(row.cy) - 64, 0, 384)); x0 = int(np.clip(round(row.cx) - 64, 0, 384))
        batch.append(img[y0:y0 + 128, x0:x0 + 128]); meta.append((r, y0, x0))
        if len(batch) == 32:
            flush()
    flush()
    return out


def stage2_masks(P, thr, T, probs, n, t2, r):
    """Build final 512x512 masks: each kept blob replaced by its refined version (None = stage-1 only)."""
    t8 = int(round(thr * 255))
    masks = [None] * n
    for s, grp in T.groupby("s"):
        b = (P[s] >= t8).astype(np.uint8)
        _, lab = cv2.connectedComponents(b, connectivity=8)
        m = np.zeros((512, 512), bool)
        for idx, row in zip(grp.index, grp.itertuples()):
            comp = lab == row.k
            if probs is None:
                m |= comp; continue
            y0, x0, pr = probs[idx]
            region = comp[y0:y0 + 128, x0:x0 + 128].astype(np.uint8)
            if r:
                region = cv2.dilate(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
            ref = (pr >= t2) & region.astype(bool)
            if ref.sum() == 0:
                m |= comp
            else:
                m[y0:y0 + 128, x0:x0 + 128] |= ref
                m &= ~(comp & ~np.pad(ref, ((y0, 384 - y0), (x0, 384 - x0))))
        masks[s] = m
    return masks


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    D = {}
    for split in ("val", "test"):
        files = pd.read_csv(CACHE / f"{split}_files.csv").file.tolist()
        G = np.load(CACHE / f"{split}_gt.npy", mmap_mode="r")
        D[split] = dict(files=files, G=G, L=lung_cache(split, files), gsum=G.reshape(len(G), -1).sum(1).astype(float),
                        NA=np.load(CACHE / f"{split}_nb_above.npy", mmap_mode="r"),
                        NB=np.load(CACHE / f"{split}_nb_below.npy", mmap_mode="r"),
                        P={src: np.load(CACHE / f"{split}_{src}.npy", mmap_mode="r") for src in ("raw", "tta")})
        log(f"{split}: {len(files)} slices, lung fields ready ({time.time() - t0:.0f}s)")

    tables = {}
    for split, src, thr in itertools.product(("val", "test"), ("raw", "tta"), THRS):
        d = D[split]
        tables[split, src, thr] = comp_table(d["P"][src], d["G"], d["L"], d["NA"], d["NB"], thr)
    log(f"component tables built ({time.time() - t0:.0f}s)")

    def best(srcs, thrs, areas, lungs, cons):
        res = []
        for src, thr, a, l, c in itertools.product(srcs, thrs, areas, lungs, cons):
            T = tables["val", src, thr]
            res.append((metrics_from(T, keep_mask(T, a, l, c), D["val"]["gsum"], len(D["val"]["files"]))["dice_global"],
                        dict(src=src, thr=thr, area=a, lung=l, cons=c)))
        return max(res, key=lambda z: z[0])

    steps = [("A. Raw model (threshold 0.5)", (["raw"], [0.5], [0], [None], [None])),
             ("B. + threshold tuned on val", (["raw"], THRS, [0], [None], [None])),
             ("C. + lung mask + tiny-blob removal", (["raw"], THRS, AREAS, LUNG_D, [None])),
             ("D. + slice-consistency check", (["raw"], THRS, AREAS, LUNG_D, CONS)),
             ("E. + test-time augmentation (Phase 1 complete)", (["raw", "tta"], THRS, AREAS, LUNG_D, CONS))]
    rows, cfgs = [], {}
    for name, grid in steps:
        vdice, cfg = best(*grid)
        out = {}
        for split in ("val", "test"):
            T = tables[split, cfg["src"], cfg["thr"]]
            out[split] = metrics_from(T, keep_mask(T, cfg["area"], cfg["lung"], cfg["cons"]), D[split]["gsum"], len(D[split]["files"]))
        rows.append(dict(step=name, **{f"val_{k}": v for k, v in out["val"].items()}, **{f"test_{k}": v for k, v in out["test"].items()}))
        cfgs[name] = cfg
        log(f"{name}: cfg {cfg} | val dice {out['val']['dice_global']:.4f} | test dice {out['test']['dice_global']:.4f} "
            f"fp {out['test']['fp_slice_rate']:.3f}")

    # ---------------- stage 2 on top of the Phase-1 configuration
    cfg = cfgs[steps[-1][0]]
    crop = load_crop_model()
    S2 = {}
    for split in ("val", "test"):
        d = D[split]
        T = tables[split, cfg["src"], cfg["thr"]]
        T = T[keep_mask(T, cfg["area"], cfg["lung"], cfg["cons"])].reset_index(drop=True)
        S2[split] = (T, crop_probs(crop, d["files"], T))
        log(f"stage 2 patches {split}: {len(T)} ({time.time() - t0:.0f}s)")
    grid = []
    T, pr = S2["val"]
    Pv = D["val"]["P"][cfg["src"]]
    for t2, r in itertools.product([0.3, 0.4, 0.5, 0.6, 0.7], [0, 3, 6, 10, 20]):
        m = metrics_masks(stage2_masks(Pv, cfg["thr"], T, pr, len(Pv), t2, r), D["val"]["G"])
        grid.append((m["dice_global"], t2, r))
        log(f"  stage2 val t2={t2} r={r}: dice {m['dice_global']:.4f}")
    _, t2, r = max(grid)
    out = {}
    for split in ("val", "test"):
        T, pr = S2[split]
        P = D[split]["P"][cfg["src"]]
        out[split] = metrics_masks(stage2_masks(P, cfg["thr"], T, pr, len(P), t2, r), D[split]["G"])
    name = "F. + two-stage refinement (crop model)"
    rows.append(dict(step=name, **{f"val_{k}": v for k, v in out["val"].items()}, **{f"test_{k}": v for k, v in out["test"].items()}))
    cfgs[name] = dict(**cfg, stage2_threshold=t2, stage2_dilate=r)
    log(f"{name}: t2={t2} r={r} | val dice {out['val']['dice_global']:.4f} | test dice {out['test']['dice_global']:.4f} "
        f"fp {out['test']['fp_slice_rate']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "final_results.csv", index=False)
    json.dump(cfgs, open(OUT / "pipeline_config.json", "w"), indent=2)
    log("\nTEST SET (105 patients, 2,558 full 512x512 slices)")
    log(df[["step", "test_dice_global", "test_dice_pos", "test_iou_global", "test_precision", "test_recall",
            "test_fp_slice_rate"]].round(4).to_string(index=False))
    log(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
