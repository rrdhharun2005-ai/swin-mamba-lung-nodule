"""Build 2.5D inputs for the existing processed dataset (needs the raw DICOMs in dataset/LIDC-IDRI).

For every slice listed in processed/metadata.csv, writes processed/images25d/<file>.png as a
3-channel image: (slice above, slice itself, slice below), each in the same lung window.
Neighbours are taken ~2 mm away (k = max(1, round(2.0 / slice_spacing))) so every scan gets the
same physical context regardless of slice thickness. Masks, slice choice and splits are unchanged.

Run with the system Python (pylidc):  python code/make_25d.py [--workers 6]
"""
import argparse
import configparser
import csv
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

configparser.SafeConfigParser = configparser.ConfigParser
for _a, _t in [("int", int), ("float", float), ("bool", bool)]:
    if not hasattr(np, _a):
        setattr(np, _a, _t)

import cv2  # noqa: E402
import pylidc as pl  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "dataset"
RAW, OUT = ROOT / "LIDC-IDRI", ROOT / "processed"
HU_MIN, HU_MAX, NEIGHBOUR_MM = -1000, 400, 2.0


def to_uint8(hu):
    hu = np.clip(hu, HU_MIN, HU_MAX).astype(np.float32)
    return ((hu - HU_MIN) / (HU_MAX - HU_MIN) * 255).astype(np.uint8)


def process_patient(args):
    pid, items = args  # items: [(file, z), ...]
    todo = [(f, z) for f, z in items if not (OUT / "images25d" / f).exists()]
    if not todo:
        return pid, 0, None
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
    vol = scan.to_volume(verbose=False)
    Z = vol.shape[2]
    k = max(1, round(NEIGHBOUR_MM / float(scan.slice_spacing)))
    # sanity check: centre channel must equal the existing 2D image
    f0, z0 = todo[0]
    old = cv2.imread(str(OUT / "images" / f0), cv2.IMREAD_GRAYSCALE)
    if not np.array_equal(old, to_uint8(vol[:, :, z0])):
        return pid, 0, "centre slice does not match existing image"
    for f, z in todo:
        up, down = vol[:, :, max(z - k, 0)], vol[:, :, min(z + k, Z - 1)]
        rgb = np.stack([to_uint8(up), to_uint8(vol[:, :, z]), to_uint8(down)], -1)
        cv2.imwrite(str(OUT / "images25d" / f), rgb[:, :, ::-1])  # stored so channel order reads (up, centre, down)
    return pid, len(todo), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    (OUT / "images25d").mkdir(exist_ok=True)
    by_pid = defaultdict(list)
    with open(OUT / "metadata.csv") as f:
        for r in csv.DictReader(f):
            by_pid[r["patient_id"]].append((r["file"], int(r["slice"])))
    missing = [p for p in by_pid if not (RAW / p).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} patients have no raw DICOMs, e.g. {missing[:3]}")

    done, errors = 0, []
    with Pool(a.workers) as pool:
        for i, (pid, n, err) in enumerate(pool.imap_unordered(process_patient, sorted(by_pid.items())), 1):
            done += n
            if err:
                errors.append((pid, err))
            if i % 50 == 0 or i == len(by_pid):
                print(f"{i}/{len(by_pid)} patients, {done} images written", flush=True)
    if errors:
        print("ERRORS:", errors)
        raise SystemExit(1)
    print("Done.")


if __name__ == "__main__":
    main()
