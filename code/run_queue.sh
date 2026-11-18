#!/usr/bin/env bash
# Queue for the comparative study. Run from the CV folder:  bash code/run_queue.sh
# Waits for the 2.5D Swin-Mamba run, then trains + evaluates U-Net 2D and Swin-UNet 2D, one after another.
# Each step waits for the previous run's "done." line, so if a run is stopped, fixed and restarted,
# the queue simply keeps waiting for the restarted run to finish.
# All checkpoints/metrics are kept in runs/<name>/ (best.pt, last.pt, log.csv); test results in runs/eval_<name>_test.txt
cd "$(dirname "$0")/.." || exit 1
PY=.venv/Scripts/python

COOLDOWN=${COOLDOWN:-1800}  # seconds of GPU rest between training runs (laptop-friendly)

wait_done() {  # $1 = training log
  until grep -q "^done\. best val dice" "$1" 2>/dev/null; do sleep 60; done
  sleep 20  # let the process exit and release the GPU
}

cooldown() {
  echo "$(date) cooling down ${COOLDOWN}s"
  sleep "$COOLDOWN"
}

wait_done runs/train_swin_mamba_25d.log
$PY code/evaluate.py --ckpt runs/swin_mamba_25d/best.pt --vis 20 > runs/eval_swin_mamba_25d_test.txt 2>&1
cooldown

$PY code/run_baseline.py train --model unet --name unet_2d --epochs 60 --batch-size 16 --workers 4 \
  > runs/train_unet_2d.log 2>&1
wait_done runs/train_unet_2d.log
$PY code/run_baseline.py eval --model unet --ckpt runs/unet_2d/best.pt --vis 20 > runs/eval_unet_2d_test.txt 2>&1
cooldown

$PY code/run_baseline.py train --model swin_unet --name swin_unet_2d --epochs 60 --batch-size 16 --workers 4 \
  > runs/train_swin_unet_2d.log 2>&1
wait_done runs/train_swin_unet_2d.log
$PY code/run_baseline.py eval --model swin_unet --ckpt runs/swin_unet_2d/best.pt --vis 20 > runs/eval_swin_unet_2d_test.txt 2>&1

echo "QUEUE FINISHED"
