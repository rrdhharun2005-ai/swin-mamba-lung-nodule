"""Train / evaluate a baseline (U-Net, Swin-UNet) with EXACTLY the same pipeline as the Swin-Mamba model.

It reuses train.py / evaluate.py unchanged (same data, augmentation, loss, optimizer, schedule, metrics)
and only swaps the network class, so the comparison is fair.

Usage (from the CV folder):
    .venv/Scripts/python code/run_baseline.py train --model unet --name unet_2d --epochs 60 --batch-size 16
    .venv/Scripts/python code/run_baseline.py train --model swin_unet --name swin_unet_2d --epochs 60
    .venv/Scripts/python code/run_baseline.py eval  --model unet --ckpt runs/unet_2d/best.pt --vis 20
Any other train.py / evaluate.py flag (e.g. --in-ch 3, --lr, --resume) is passed through.
"""
import sys

from baselines import MODELS


def main():
    if len(sys.argv) < 4 or sys.argv[1] not in ("train", "eval") or "--model" not in sys.argv:
        raise SystemExit(__doc__)
    mode = sys.argv[1]
    i = sys.argv.index("--model")
    name = sys.argv[i + 1]
    if name not in MODELS:
        raise SystemExit(f"unknown model {name!r}; choose from {list(MODELS)}")
    sys.argv = [sys.argv[0]] + sys.argv[2:i] + sys.argv[i + 2:]  # strip mode and --model for the inner parser

    if mode == "train":
        import train as target
    else:
        import evaluate as target
    target.SwinMambaUNet = MODELS[name]  # swap the network; everything else is identical
    target.main()


if __name__ == "__main__":
    main()
