import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from PIL import Image

OUT = Path(sys.argv[1])
DATA = Path(r"D:\Claude\CV\dataset\processed")
SAMPLE = "LIDC-IDRI-0194_073.png"
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11, "mathtext.fontset": "stix"})

DARK, GREY, ORANGE = "#262626", "#7F7F7F", "#E8891C"
STEM, SWIN, VSS, DEC, MERGE, LN, HEAD = "#DCE9F7", "#E2DCF2", "#DDEFD9", "#FBE3CF", "#EDEDED", "#E4F2DC", "#DCE9F7"
EDGE = {STEM: "#4F81BD", SWIN: "#5B4A9B", VSS: "#3C7A3A", DEC: "#C0712B", MERGE: "#595959", LN: "#3C7A3A"}

W, H = 13.0, 10.6
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def rbox(x, y, w, h, fc, ec, lw=1.2, r=0.08, ls="-", z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                linestyle=ls, zorder=z))


def label(x, y, t, fs=11, **k):
    ax.text(x, y, t, ha="center", va="center", fontsize=fs, zorder=5, **k)


def two(x, y, w, top, bot, fc_top, fc_bot=MERGE, h=0.42):
    """two-part block like 'Mixer x2 / Patch Merging': bottom part then top part"""
    rbox(x, y, w, 2 * h, "white", DARK, lw=1.0, r=0.06)
    rbox(x, y + h, w, h, fc_top, EDGE.get(fc_top, DARK), lw=1.0, r=0.06, z=3)
    rbox(x, y, w, h, fc_bot, EDGE.get(fc_bot, DARK), lw=1.0, r=0.06, z=3)
    label(x + w / 2, y + 1.5 * h - (0.06 if "Attention" in top else 0), top); label(x + w / 2, y + 0.5 * h, bot)


def one(x, y, w, t, fc, h=0.42, fs=11):
    rbox(x, y, w, h, fc, EDGE.get(fc, DARK), lw=1.0, r=0.06, z=3)
    label(x + w / 2, y + h / 2, t, fs=fs)


def arr(p, q, color=DARK, lw=1.2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=4))


def line(xs, ys, color=DARK, lw=1.2):
    ax.plot(xs, ys, color=color, lw=lw, zorder=4, solid_capstyle="butt")


def concat(x, y, r=0.15):
    ax.add_patch(Circle((x, y), r, fc="white", ec=DARK, lw=1.2, zorder=5))
    label(x, y - 0.005, "C", fs=10, fontweight="bold")


def dim(x, y, _, s, c):
    ax.text(x, y, rf"$\frac{{H}}{{{s}}}\times\frac{{W}}{{{s}}}\times {c}$", ha="center", va="center", fontsize=12.5,
            color="#555555", zorder=5)


# ---------------- geometry
EX, EW = 0.75, 2.6          # encoder column
DX, DW = 9.65, 2.6          # decoder column
ecx, dcx = EX + EW / 2, DX + DW / 2
LX, LW = 6.1, 1.5          # skip LayerNorm boxes
h = 0.42
b1, b2, b3, b4 = 5.25, 3.85, 2.45, 1.15        # encoder pair bottoms (b4 = single bottleneck box)
f1, f2, f3 = b1 + 1.5 * h, b2 + 1.5 * h, b3 + 1.5 * h
f4 = b4 + h / 2
stem_y = 6.75; f0 = stem_y + h / 2

# groups (dashed)
rbox(EX - 0.3, 0.92, EW + 0.6, 5.68, "#F7F7F7", "#555", lw=1.3, r=0.35, ls=(0, (5, 3)), z=1)
rbox(5.2, 2.85, 2.75, 4.45, "#F2F2F2", "#555", lw=1.3, r=0.35, ls=(0, (5, 3)), z=1)
rbox(DX - 0.3, 1.8, DW + 0.6, 6.05, "#F7F7F7", "#555", lw=1.3, r=0.35, ls=(0, (5, 3)), z=1)
ax.text(EX - 0.25, 0.75, "Encoder", fontsize=11, style="italic", color="#555")
ax.text(5.25, 2.65, "Skip connections", fontsize=11, style="italic", color="#555")
ax.text(DX + DW + 0.3, 1.63, "Decoder", fontsize=11, style="italic", color="#555", ha="right")

# input / output thumbnails
img = np.array(Image.open(DATA / "images" / SAMPLE).convert("L"))
msk = np.array(Image.open(DATA / "masks" / SAMPLE).convert("L")) > 127
ax.imshow(img, cmap="gray", extent=(ecx - 0.7, ecx + 0.7, 8.75, 10.15), zorder=3)
for dx in (0.12, 0.06):
    ax.add_patch(plt.Rectangle((ecx - 0.7 + dx, 8.75 + dx), 1.4, 1.4, fc="#333", ec="white", lw=0.6, zorder=2))
ax.text(ecx - 0.85, 9.6, "$H\\times W\\times 1$ (2D)\n$H\\times W\\times 3$ (2.5D)", ha="right", va="center", fontsize=12, color="#555")
ov = np.stack([img] * 3, -1).astype(float) / 255
ov[msk] = [0.85, 0.05, 0.1]
ax.imshow(ov, extent=(dcx - 0.7, dcx + 0.7, 8.75, 10.15), zorder=3)
ax.text(dcx + 0.85, 9.45, "$H\\times W\\times 1$\nnodule mask", ha="left", va="center", fontsize=12, color="#555")

# stem
one(EX, stem_y, EW, "Conv Stem (2 × Conv 3×3, s2)", STEM)
arr((ecx, 8.75), (ecx, stem_y + h))
# encoder blocks
two(EX, b1, EW, "Swin Attention Block × 2", "Patch Merging", SWIN)
two(EX, b2, EW, "Swin Attention Block × 2", "Patch Merging", SWIN)
# highlight where attention is used
AC = "#C0392B"
rbox(EX - 0.12, b2 - 0.1, EW + 0.24, (b1 + 2 * h) - b2 + 0.2, "none", "none", lw=0, r=0.12, z=1.5)
rbox(EX - 0.12, b2 - 0.1, EW + 0.24, (b1 + 2 * h) - b2 + 0.2, "none", AC, lw=2.4, r=0.12, ls=(0, (1.5, 1.5)), z=6)
# big callout in the free gap between encoder and skip group
cx, cy = 4.28, (f1 + f2 + 0.3) / 2 + 0.02
ax.text(4.02, cy + 0.11, "Attention", ha="left", va="center", fontsize=14, color=AC, fontweight="bold", zorder=9)
ax.text(4.02, cy - 0.17, "(window self-attention)", ha="left", va="center", fontsize=8.5, color=AC, zorder=9)
ax.add_patch(FancyArrowPatch((3.96, cy + 0.11), (EX + EW + 0.16, cy + 0.11), arrowstyle="-|>", mutation_scale=18, color=AC, lw=2.5, zorder=8))

for yy in (b1, b2):
    ax.text(EX + EW / 2, yy + 2 * h - 0.06, "8×8 windows · W-MSA → SW-MSA", ha="center", va="top", fontsize=9, color="#5B4A9B", zorder=6)
two(EX, b3, EW, "VSS (Mamba) Block × 2", "Patch Merging", VSS)
one(EX, b4, EW, "VSS (Mamba) Block × 2", VSS)
arr((ecx, stem_y), (ecx, b1 + 2 * h))
for top, bot in ((b1, b2), (b2, b3), (b3, b4)):
    arr((ecx, top), (ecx, bot + (2 * h if bot != b4 else h)))

# skips through the middle group
skips = [(f0, "S", "2", "16"), (f1, "1", "4", "32"), (f2, "2", "8", "64"), (f3, "3", "16", "128")]
for y, tag, s, c in skips:
    x0 = EX + EW
    line([x0, LX - 0.62], [y, y])
    ax.add_patch(FancyBboxPatch((LX - 0.62, y - 0.17), 0.45, 0.34, boxstyle="round,pad=0,rounding_size=0.05",
                                fc="#F0F0F0", ec=DARK, lw=1.0, zorder=5))
    label(LX - 0.395, y, rf"$F^{{{tag}}}$" if tag != "S" else r"$F^{0}$", fs=11)
    arr((LX - 0.17, y), (LX, y))
    if tag == "S":
        one(LX, y - 0.17, LW, "identity", MERGE, h=0.34, fs=10)
    else:
        one(LX, y - 0.17, LW, "LayerNorm", LN, h=0.34, fs=10)
    arr((LX + LW, y), (dcx - 0.15, y))
    dim(4.45, y + 0.3, None, s, c)
# bottleneck route
line([EX + EW, dcx], [f4, f4]); arr((dcx, f4), (dcx, 2.05))
dim(6.3, f4 + 0.3, None, "32", "256")

# decoder (bottom -> top)
one(DX, 2.05, DW, "Upsample × 2 (bilinear)", DEC)
arr((dcx, 2.05 + h), (dcx, f3 - 0.15))
concat(dcx, f3)
two(DX, f3 + 0.3, DW, "Upsample × 2", "Conv Block × 2 · 128", DEC, DEC, h=0.4)
line([dcx, dcx], [f3 + 0.15, f3 + 0.3]); arr((dcx, f3 + 1.1), (dcx, f2 - 0.15))
concat(dcx, f2)
two(DX, f2 + 0.3, DW, "Upsample × 2", "Conv Block × 2 · 64", DEC, DEC, h=0.4)
line([dcx, dcx], [f2 + 0.15, f2 + 0.3]); arr((dcx, f2 + 1.1), (dcx, f1 - 0.15))
concat(dcx, f1)
two(DX, f1 + 0.22, DW, "Upsample × 2", "Conv Block × 2 · 32", DEC, DEC, h=0.31)
line([dcx, dcx], [f1 + 0.15, f1 + 0.22]); arr((dcx, f1 + 0.84), (dcx, f0 - 0.15))
concat(dcx, f0)
one(DX, f0 + 0.3, DW, "Conv Block × 2 · 16", DEC, h=0.4)
line([dcx, dcx], [f0 + 0.15, f0 + 0.3])
one(DX, 8.0, DW, "Head: Conv, ×2 upsample, sigmoid", HEAD, h=0.42, fs=10.5)
arr((dcx, f0 + 0.7), (dcx, 8.0)); arr((dcx, 8.42), (dcx, 8.75))

# title
ax.text(W / 2, 10.38, "Lightweight Swin-Mamba U-Net  ·  3.08 M parameters  ·  4.85 GMACs per 512×512 slice",
        ha="center", fontsize=14, fontweight="bold", color="#820019")

# legend
rbox(0.45, 0.05, W - 0.9, 0.55, "white", "#555", lw=1.2, r=0.2, ls=(0, (5, 3)))
items = [(STEM, "Conv stem"), (SWIN, "Swin attention block (window self-attention)"), (VSS, "VSS block (Mamba SS2D, 4-way scan)"),
         (DEC, "Conv decoder block"), (MERGE, "Patch merging")]
x = 0.7
for fc, t in items:
    rbox(x, 0.19, 0.38, 0.27, fc, EDGE.get(fc, DARK), lw=1.0, r=0.04)
    ax.text(x + 0.48, 0.325, t, fontsize=9.5, va="center")
    x += 0.62 + len(t) * 0.068
ax.add_patch(Circle((x + 0.1, 0.325), 0.13, fc="white", ec=DARK, lw=1.1, zorder=5)); label(x + 0.1, 0.32, "C", fs=9, fontweight="bold")
ax.text(x + 0.32, 0.325, "Concatenation", fontsize=10, va="center")

fig.savefig(OUT, dpi=300, facecolor="white")
print("ok")
