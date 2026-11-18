"""Convert downloaded LIDC-IDRI CT scans into 2D image/mask PNG pairs for segmentation.

- Image: axial slice, HU clipped to the lung window [-1000, 400] and scaled to 0-255 (512x512).
- Mask : union of all nodules, each nodule = 50% radiologist-consensus of its annotations.
- Kept : every slice containing a nodule + an equal number of random nodule-free slices
         (taken at least 3 slices away from any nodule) per patient.
- Split: patient-level train/val/test = 70/15/15 (no patient leakage), written to metadata.csv.
- Incremental: patients already in metadata.csv are kept as-is; only newly downloaded ones are added.

Usage: python code/preprocess_lidc.py [--neg-ratio 1.0] [--workers 4]
"""
import argparse
import configparser
import csv
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np

# --- compatibility shims: pylidc predates Python 3.12 / NumPy 2 ---
configparser.SafeConfigParser = configparser.ConfigParser
for _a, _t in [("int", int), ("float", float), ("bool", bool)]:
    if not hasattr(np, _a):
        setattr(np, _a, _t)

import cv2  # noqa: E402
import pylidc as pl  # noqa: E402
from pylidc.utils import consensus  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "dataset"
RAW = ROOT / "LIDC-IDRI"
OUT = ROOT / "processed"
HU_MIN, HU_MAX = -1000, 400
SEED = 42


def to_uint8(hu):
    hu = np.clip(hu, HU_MIN, HU_MAX).astype(np.float32)
    return ((hu - HU_MIN) / (HU_MAX - HU_MIN) * 255).astype(np.uint8)


def process_patient(args):
    pid, neg_ratio = args
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
    if scan is None:
        return []
    vol = scan.to_volume(verbose=False)  # (512, 512, Z) in HU
    mask = np.zeros(vol.shape, dtype=bool)
    for anns in scan.cluster_annotations():
        cmask, cbbox, _ = consensus(anns, clevel=0.5, pad=[(512, 512), (512, 512), (0, 0)])
        mask[cbbox] |= cmask

    z_pos = [z for z in range(vol.shape[2]) if mask[:, :, z].any()]
    near = {z + d for z in z_pos for d in range(-3, 4)}
    z_neg_pool = [z for z in range(vol.shape[2]) if z not in near]
    rng = random.Random(f"{SEED}-{pid}")
    n_neg = min(len(z_neg_pool), max(1, round(len(z_pos) * neg_ratio)))
    z_neg = rng.sample(z_neg_pool, n_neg)

    rows = []
    for z, has in [(z, 1) for z in z_pos] + [(z, 0) for z in z_neg]:
        name = f"{pid}_{z:03d}.png"
        cv2.imwrite(str(OUT / "images" / name), to_uint8(vol[:, :, z]))
        cv2.imwrite(str(OUT / "masks" / name), mask[:, :, z].astype(np.uint8) * 255)
        rows.append({"file": name, "patient_id": pid, "slice": z, "has_nodule": has,
                     "nodule_pixels": int(mask[:, :, z].sum())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg-ratio", type=float, default=1.0, help="negative slices per positive slice")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "masks").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "selected_patients.csv") as f:
        pids = sorted(r["PatientID"] for r in csv.DictReader(f))

    # Incremental: keep already-processed patients (and their splits), process only new ones
    # whose raw DICOMs are on disk.
    meta = OUT / "metadata.csv"
    old_rows = []
    if meta.exists():
        with open(meta) as f:
            old_rows = list(csv.DictReader(f))
    done = {r["patient_id"] for r in old_rows}
    pids = [p for p in pids if p not in done and (RAW / p).exists()]
    if not pids:
        print("Nothing new to process.")
        return
    print(f"{len(done)} patients already processed, {len(pids)} new")

    shuffled = pids[:]
    random.Random(SEED).shuffle(shuffled)
    n = len(shuffled)
    split = {p: "train" for p in shuffled[: int(0.7 * n)]}
    split.update({p: "val" for p in shuffled[int(0.7 * n): int(0.85 * n)]})
    split.update({p: "test" for p in shuffled[int(0.85 * n):]})

    rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process_patient, [(p, a.neg_ratio) for p in pids]), 1):
            rows.extend(r)
            if i % 20 == 0 or i == len(pids):
                print(f"{i}/{len(pids)} patients, {len(rows)} slices", flush=True)

    for r in rows:
        r["split"] = split[r["patient_id"]]
    rows = old_rows + rows
    rows.sort(key=lambda r: r["file"])
    with open(OUT / "metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    pos = sum(int(r["has_nodule"]) for r in rows)
    print(f"Done: {len(rows)} slices ({pos} with nodules, {len(rows) - pos} without)")
    for s in ("train", "val", "test"):
        print(f"  {s}: {sum(r['split'] == s for r in rows)} slices")


if __name__ == "__main__":
    main()
