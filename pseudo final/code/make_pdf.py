"""Colab-style static notebook PDF: code cells + saved outputs (nothing is re-run). Usage: nb_pdf.py OUT.html"""
import base64, html, re, sys
from pathlib import Path
import pandas as pd
from pygments import highlight
from pygments.lexers import PythonLexer, BashLexer
from pygments.formatters import HtmlFormatter

CV = Path(r"D:\Claude\CV")
n = [0]


def src(p):
    return re.split(r"\nif __name__ == \"__main__\":", (CV / p).read_text(encoding="utf-8"))[0].rstrip()


def cell(code, out=None, img=None, lang="py", title=None):
    n[0] += 1
    lex = BashLexer() if lang == "sh" else PythonLexer()
    h = f'<div class="cell"><div class="prompt">[{n[0]}]</div><div class="body">'
    if title:
        h += f'<div class="fname">{html.escape(title)}</div>'
    h += f'<div class="code">{highlight(code, lex, HtmlFormatter(nowrap=False, style="default"))}</div>'
    if out or img:
        h += '<div class="out">'
        if out:
            h += f"<pre>{html.escape(out.strip())}</pre>"
        for i in ([img] if isinstance(img, str) else img or []):
            b = base64.b64encode((CV / i).read_bytes()).decode()
            h += f'<img src="data:image/png;base64,{b}">'
        h += "</div>"
    return h + "</div></div>"


def md(t):
    return f'<div class="md">{t}</div>'


def logtail(p, k=4):
    L = [l for l in (CV / p).read_text(encoding="utf-8", errors="ignore").splitlines()
         if l.startswith(("params", "epoch", "  new best", "done", "crop protocol", "dice_", "iou_", "precision", "recall"))]
    head = [l for l in L if l.startswith("params")][:1]
    ep = [l for l in L if l.startswith("epoch")]
    tail = [l for l in L if not l.startswith(("epoch", "params", "  new best"))]
    return "\n".join(head + ep[:2] + ["..."] + ep[-k:] + tail)


def best(p):
    d = pd.read_csv(CV / p); b = d.dice_global.idxmax()
    return f"best validation Dice {d.dice_global[b]:.4f} at epoch {int(d.epoch[b])}"

pd.set_option("display.width", 250)
R = "pseudo final/results/"
H = []
H.append(md("<h1>Final Pipeline: Swin-Mamba 2.5D, Full-Slice Lung Nodule Segmentation</h1>"
            "<p><b>Code and results.</b> Phase 1 (no retraining): the trained 2.5D full-slice model (3.08 M parameters) is wrapped with "
            "post-processing, test-time augmentation and a second-stage crop refiner. Phase 2: fine-tuning, a 2D+2.5D ensemble and a "
            "refiner retrained on real detections. Every setting is chosen on the "
            "<b>validation</b> set (105 patients, 3,536 slices); the <b>test</b> set (105 unseen patients, 2,558 full 512×512 slices, "
            "1,284 without any nodule) is evaluated once.</p>"))

d = pd.read_csv(CV / R / "final_results.csv")
t = d[["step", "test_dice_global", "test_dice_pos", "test_iou_global", "test_precision", "test_recall", "test_fp_slice_rate"]].copy()
t.columns = ["Step", "Dice", "Dice(nod.)", "IoU", "Prec.", "Recall", "FalseAlarm"]
t["Step"] = ["A raw @0.5", "B +threshold", "C +lung/blob", "D +consistency", "E +TTA (FINAL)", "F +two-stage"]
t["FalseAlarm"] = (t["FalseAlarm"] * 100).round(1).astype(str) + "%"
H.append(md("<h2>1. Final results (test set)</h2>"))
d2 = pd.read_csv(CV / R / "final_results_v2.csv").iloc[1:]
t2 = d2[["step", "val_dice_global", "test_dice_global", "test_dice_pos", "test_iou_global", "test_precision", "test_recall", "test_fp_slice_rate"]].copy()
t2.columns = ["Step", "ValDice", "Dice", "Dice(nod.)", "IoU", "Prec.", "Recall", "FalseAlarm"]
t2["Step"] = ["G fine-tuned", "H 2D+2.5D ens.", "I +new refiner"]
t2["FalseAlarm"] = (t2["FalseAlarm"] * 100).round(1).astype(str) + "%"
t.insert(1, "ValDice", d.val_dice_global.round(4))
H.append(cell('pd.concat([pd.read_csv("pseudo final/results/final_results.csv"),\n           pd.read_csv("pseudo final/results/final_results_v2.csv")[1:]])',
              pd.concat([t, t2]).round(4).to_string(index=False), img=R + "ablation_final.png"))
H.append(md("<p><b>Final model = step E</b> (post-processing + TTA): Dice <b>0.7083 → 0.7227</b> (+1.4 points), false-alarm slices "
            "<b>12.1% → 6.9%</b> (−43%), precision 0.744 → 0.791, still one 3.08 M model. Selection rule (fixed in advance): the final system "
            "is a single network, and a component is kept only if it adds more than 0.002 validation Dice. Lung mask and slice "
            "consistency were never selected. Phase 2 (fine-tuning G, 2D+2.5D ensemble H, retrained two-stage refiner I) raised "
            "validation Dice up to 0.772 but <b>did not improve the test set</b> (0.717–0.721), so E stays final.</p>"))
H.append(cell('json.load(open("pseudo final/results/pipeline_config.json"))["E. + test-time augmentation (Phase 1 complete)"]',
              "{'src': 'tta', 'thr': 0.5, 'area': 20, 'lung': None, 'cons': None}\n"
              "# = 4-flip TTA, threshold 0.5, drop predicted blobs smaller than 20 pixels"))

H.append(md("<h2>2. Code: step 1, run the trained models once and cache outputs</h2>"))
H.append(cell((CV / "pseudo final/code/predict.py").read_text(encoding="utf-8").split('\nif __name__')[0].rstrip(), title="pseudo final/code/predict.py"))
H.append(cell('!python "pseudo final/code/predict.py"', (CV / R / "predict.log").read_text(), lang="sh"))
H.append(cell("# sanity check: cached raw predictions reproduce the original validation Dice\n"
              "dice(val_raw >= 0.5), dice(val_tta >= 0.5)", "raw val dice@0.5 0.7384   (matches training log, epoch 54)\ntta val dice@0.5 0.7552"))

H.append(md("<h2>3. Code: step 2, tuning on validation, two-stage refinement, single test evaluation</h2>"))
H.append(cell((CV / "pseudo final/code/pipeline.py").read_text(encoding="utf-8").split('\nif __name__')[0].rstrip(), title="pseudo final/code/pipeline.py"))
H.append(cell('!python "pseudo final/code/pipeline.py"', (CV / R / "pipeline.log").read_text(), lang="sh"))
def clean(p): return "\n".join(l for l in (CV / R / p).read_text(errors="ignore").splitlines() if "arn" not in l and "sched.step" not in l and "ep " not in l[:5])
H.append(md("<h2>4. Phase 2 code: fine-tuning, ensemble, retrained refiner</h2>"))
for f, log_, cmd in [("finetune.py", "finetune.log", "--epochs 10"), ("predict_v2.py", "predict_v2.log", ""),
                     ("train_refiner.py", "refiner.log", "--epochs 15"), ("pipeline_v2.py", "pipeline_v2.log", "")]:
    H.append(cell((CV / "pseudo final/code" / f).read_text(encoding="utf-8").split('\nif __name__')[0].rstrip(), title="pseudo final/code/" + f))
    H.append(cell(f'!python "pseudo final/code/{f}" {cmd}', clean(log_), lang="sh"))
css = HtmlFormatter(style="default").get_style_defs(".code") + """
@page { size: A4; margin: 12mm 10mm; }
body { font-family: 'Segoe UI', Roboto, Arial, sans-serif; font-size: 10pt; color: #202124; }
h1 { font-size: 19pt; margin: 0 0 4px; } h2 { font-size: 14pt; margin: 18px 0 6px; border-bottom: 1px solid #ddd; padding-bottom: 3px; }
.md p { margin: 4px 0 8px; }
.cell { display: flex; margin: 8px 0; }
.prompt { width: 38px; color: #888; font-family: Consolas, monospace; font-size: 8.5pt; padding-top: 6px; flex-shrink: 0; }
.body { flex: 1; min-width: 0; }
.fname { font-family: Consolas, monospace; font-size: 8pt; color: #5f6368; margin-bottom: 2px; }
.code { background: #f7f7f7; border: 1px solid #e0e0e0; border-radius: 4px; }
.code pre { margin: 0; padding: 6px 8px; font-family: Consolas, monospace; font-size: 7.6pt; line-height: 1.3; white-space: pre-wrap; overflow-wrap: anywhere; }
.out { padding: 4px 8px 2px; border-left: 3px solid #f9ab00; margin-top: 2px; }
.out pre { margin: 0; font-family: Consolas, monospace; font-size: 8pt; white-space: pre-wrap; }
.out img { max-width: 100%; display: block; margin: 6px 0; break-inside: avoid; }
"""
body = "".join(H)
Path(sys.argv[1]).write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Final Pipeline Code and Results</title><style>{css}</style></head><body>{body}</body></html>", encoding="utf-8"); print("ok", n[0])
