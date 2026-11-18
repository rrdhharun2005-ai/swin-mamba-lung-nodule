"""Build the final deck from the full-slice deck: restyled literature review and thank-you slides,
'Methodology' titles, updated results with the final inference pipeline (numbers read from results/).

Usage (from D:/Claude/CV): .venv/Scripts/python "pseudo final/code/make_ppt.py"
"""
import copy
from pathlib import Path

import pandas as pd
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
CV = ROOT.parent
SRC = CV / "presentation" / "Lightweight_SwinMamba_Lung_Nodule_Segmentation.pptx"
OUT = ROOT / "Lightweight_SwinMamba_Lung_Nodule_Segmentation_FINAL.pptx"
RES = ROOT / "results"

p = Presentation(str(SRC))
S = list(p.slides)
# NEW_ARCH: architecture figure with the attention stages highlighted
_arch = (CV / "presentation" / "figures" / "fig_swinmamba.png").read_bytes()
for _s in S:
    _title = " ".join(sh.text_frame.text for sh in _s.shapes if sh.name == "Text 4" and sh.has_text_frame)
    if "Swin-Mamba U-Net" in _title:
        for _sh in _s.shapes:
            if _sh.shape_type == 13:
                _s.part.related_part(_sh._element.blip_rId)._blob = _arch
                print("arch figure replaced on:", _title)


def shape(slide, name):
    return next(sh for sh in slide.shapes if sh.name == name)


def set_text(tf, lines, size=None, bold_prefix=True, bullet=False):
    """Replace a text frame's paragraphs, keeping the first run's formatting. lines: str or (bold, rest)."""
    p0 = tf.paragraphs[0]._p
    tmpl = copy.deepcopy(p0)
    for para in list(tf.paragraphs):
        para._p.getparent().remove(para._p)
    body = tf._txBody
    for ln in lines:
        pe = copy.deepcopy(tmpl)
        runs = pe.findall(qn("a:r"))
        for r in runs[1:]:
            pe.remove(r)
        r0 = runs[0] if runs else None
        parts = [ln] if isinstance(ln, str) else [ln[0], ln[1]]
        for k, txt in enumerate(parts):
            r = copy.deepcopy(r0)
            r.find(qn("a:t")).text = txt
            rpr = r.find(qn("a:rPr"))
            if rpr is None:
                rpr = etree.SubElement(r, qn("a:rPr")); r.insert(0, rpr)
            rpr.set("b", "1" if (not isinstance(ln, str) and k == 0) else "0")
            if size:
                rpr.set("sz", str(size))
            if k == 0:
                pe.remove(r0)
            pe.append(r)
        body.append(pe)


def fill(cell, rgb):
    cell.fill.solid(); cell.fill.fore_color.rgb = RGBColor.from_string(rgb)


# ---------------------------------------------------------------- titles: "Proposed Methodology" -> "Methodology"
for s in S:
    for sh in s.shapes:
        if sh.has_text_frame:
            for para in sh.text_frame.paragraphs:
                for r in para.runs:
                    if "Proposed Methodology" in r.text:
                        r.text = r.text.replace("Proposed Methodology", "Methodology")

# ---------------------------------------------------------------- literature review (slide 4), table style as requested
s = S[3]
for sh in list(s.shapes):
    if sh.has_table or sh.name == "Text 1":
        sh._element.getparent().remove(sh._element)
rows = [
    ("Paper", "Method", "Limitation / Research Gap"),
    ("S3TU-Net: Structured Convolution and Superpixel Transformer for Lung Nodule Segmentation (2024)",
     "Structured convolutions with a superpixel transformer on LIDC-IDRI. DSC 89.0% on 128×128 patches centred on each nodule.",
     "Nodule position is given in advance and the split is 9:1 by slice. No full-slice testing and no false-alarm measure."),
    ("RAD-UNet: An Improved Lung Nodule Semantic Segmentation Algorithm Based on Deep Learning (Frontiers in Oncology, 2023)",
     "Residual U-Net with attention on 512×512 LIDC-IDRI slices. mIoU 87.8%.",
     "Only slices that contain a nodule are tested, and mIoU includes the background class. Heavy CNN."),
    ("Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation (ECCV Workshops, 2022)",
     "U-shaped network built only from Swin Transformer blocks, ImageNet pre-trained.",
     "41 M parameters and 4×4 patches that blur very small nodules. Not designed for lung CT."),
    ("LightM-UNet: Mamba Assists in Lightweight UNet for Medical Image Segmentation (2024)",
     "Lightweight U-Net with Mamba blocks for global context at linear cost.",
     "Tested on liver and chest X-ray data, not lung nodules. No 2.5D context and no comparison under one pipeline."),
    ("Cross-Modal Interactive Perception Network with Mamba for Lung Tumor Segmentation in PET-CT Images (2025)",
     "Mamba-based fusion of PET and CT on 512×512 full slices (PCLT20K).",
     "Needs PET as well as CT, segments large tumours rather than small nodules, and trains on 4× RTX 4090 GPUs."),
]
gt = s.shapes.add_table(len(rows), 3, Inches(0.45), Inches(1.45), Inches(12.45), Inches(5.3))
gt.name = "LitTable"
tblPr = gt._element.graphic.graphicData.tbl.tblPr
for k in list(tblPr.attrib):
    del tblPr.attrib[k]
sid = tblPr.find(qn("a:tableStyleId"))
if sid is not None:
    tblPr.remove(sid)
t = gt.table
for c, w in zip(t.columns, (4.35, 4.15, 3.95)):
    c.width = Inches(w)
for i, row in enumerate(rows):
    t.rows[i].height = Inches(0.45 if i == 0 else 0.97)
    for j, v in enumerate(row):
        cell = t.cell(i, j)
        fill(cell, "4472C4" if i == 0 else ("CFD8EB" if i % 2 else "E9EDF5"))
        cell.margin_left = cell.margin_right = Inches(0.1)
        cell.margin_top = cell.margin_bottom = Inches(0.05)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf = cell.text_frame; tf.word_wrap = True
        tf.text = v
        para = tf.paragraphs[0]; para.alignment = PP_ALIGN.LEFT
        run = para.runs[0]
        run.font.name = "Calibri"; run.font.size = Pt(14 if i == 0 else 12.5)
        run.font.bold = i == 0
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if i == 0 else RGBColor(0x26, 0x26, 0x26)
        # thin white borders between cells
        tcPr = cell._tc.get_or_add_tcPr()
        for side in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
            ln = etree.SubElement(tcPr, qn(side), w="12700")
            sf = etree.SubElement(ln, qn("a:solidFill")); etree.SubElement(sf, qn("a:srgbClr"), val="FFFFFF")
        tcPr.insert(0, tcPr[-4]); tcPr.insert(1, tcPr[-3]); tcPr.insert(2, tcPr[-2]); tcPr.insert(3, tcPr[-1])

# ---------------------------------------------------------------- thank-you slide (last), WordArt-style
s = S[-1]
for sh in list(s.shapes):
    if sh.name != "Text 3":
        sh._element.getparent().remove(sh._element)


def wordart(slide, text, x, y):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(4.0), Inches(1.3))
    tf = tb.text_frame; tf.word_wrap = False
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    r = tf.paragraphs[0].add_run(); r.text = text
    r.font.name = "Times New Roman"; r.font.size = Pt(66); r.font.bold = True
    rPr = r._r.get_or_add_rPr()
    for tag in ("a:solidFill",):
        for e in rPr.findall(qn(tag)):
            rPr.remove(e)
    grad = etree.fromstring(
        '<a:gradFill xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" rotWithShape="1">'
        '<a:gsLst><a:gs pos="0"><a:srgbClr val="6F9BE0"/></a:gs><a:gs pos="50000"><a:srgbClr val="2F5BB7"/></a:gs>'
        '<a:gs pos="100000"><a:srgbClr val="1F3F8F"/></a:gs></a:gsLst><a:lin ang="5400000" scaled="0"/></a:gradFill>')
    eff = etree.fromstring(
        '<a:effectLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        '<a:reflection blurRad="6350" stA="55000" endA="300" endPos="45500" dir="5400000" sy="-100000" algn="bl" rotWithShape="0"/>'
        '</a:effectLst>')
    ln = rPr.find(qn("a:ln"))
    pos = 0 if ln is None else list(rPr).index(ln) + 1
    rPr.insert(pos, grad); rPr.insert(pos + 1, eff)
    tb.name = f"WordArt {text}"
    return tb


wordart(s, "THANK", 4.65, 2.55)
wordart(s, "YOU", 6.5, 3.55)

# ---------------------------------------------------------------- results: add the final pipeline
r1 = pd.read_csv(RES / "final_results.csv")
r2 = pd.read_csv(RES / "final_results_v2.csv") if (RES / "final_results_v2.csv").exists() else None
rows_all = pd.concat([r1, r2]) if r2 is not None else r1
# final configuration, chosen on VALIDATION only: the simplest step (earliest in the list) whose validation Dice is
# within 0.002 of the best one; an extra component must earn more than that to be kept
# and the final system must stay a single 3.08 M model (lightweight claim), so ensembles / two-model pipelines are excluded
rows_all = rows_all.reset_index(drop=True)
single = rows_all[~rows_all.step.str.contains("two-stage|ensemble|refiner")]
final = single[single.val_dice_global >= single.val_dice_global.max() - 0.002].iloc[0]
raw = r1.iloc[0]

s = S[11]  # results table slide: retitle note and key findings
for sh in s.shapes:
    if sh.name == "Text 3":
        set_text(sh.text_frame, [
            ("Raw model output: ", "the 2.5D model is best on Dice, IoU, recall and false alarms."),
            ("Final pipeline (next slide): ", f"Dice {final.test_dice_global:.3f}, false alarms {final.test_fp_slice_rate * 100:.1f}%."),
            ("2D vs 2D: ", "U-Net leads by 0.027 Dice but needs 10× params, 40× compute."),
            "Bold = best per row."], size=1500)

# new slide after the qualitative slide: clone layout of the curves slide
src = S[12]
new = p.slides.add_slide(src.slide_layout)
for sh in list(new.shapes):
    sh._element.getparent().remove(sh._element)
for sh in src.shapes:
    if sh.shape_type != 13:  # skip pictures
        new.shapes._spTree.append(copy.deepcopy(sh._element))
for sh in new.shapes:
    if sh.name == "Text 4" and sh.has_text_frame:
        sh.text_frame.paragraphs[0].runs[0].text = "Results: Final Inference Pipeline"
        for r in sh.text_frame.paragraphs[0].runs[1:]:
            r.text = ""
new.shapes.add_picture(str(RES / ("ablation_final.png" if (RES / "ablation_final.png").exists() else "ablation.png")),
                       Inches(0.6), Inches(1.3), width=Inches(6.0))
tb = new.shapes.add_textbox(Inches(6.9), Inches(1.4), Inches(6.0), Inches(5.0)); tb.name = "Findings"
tf = tb.text_frame; tf.word_wrap = True
lines = [("Same trained model, no change to the network: ", "settings tuned on validation, test run once."),
         ("Test-time augmentation: ", "average over 4 flipped copies of each slice."),
         ("Tiny-blob removal: ", "drop predicted spots below a tuned size."),
         ("Result: ", f"Dice {raw.test_dice_global:.3f} → {final.test_dice_global:.3f}; false alarms "
                      f"{raw.test_fp_slice_rate * 100:.1f}% → {final.test_fp_slice_rate * 100:.1f}%; precision "
                      f"{raw.test_precision:.3f} → {final.test_precision:.3f}."),
         ("Tried, not adopted: ", "lung mask, slice consistency, fine-tuning (test 0.721), 2D+2.5D ensemble (0.717) and a retrained two-stage refiner (0.717): validation gains did not carry over to test.")]
for k, (a, b) in enumerate(lines):
    para = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
    para.space_after = Pt(9)
    for txt, bold in ((a, True), (b, False)):
        r = para.add_run(); r.text = txt; r.font.bold = bold; r.font.size = Pt(16); r.font.name = "Times New Roman"
for sh in new.shapes:
    if sh.name == "Text 5":
        sh.text_frame.paragraphs[0].runs[0].text = ("Ablation on the test set (105 patients, 2,558 slices). "
                                                     "Each step adds to the previous one.")
        for r in sh.text_frame.paragraphs[0].runs[1:]:
            r.text = ""
lst = p.slides._sldIdLst
el = lst[-1]; lst.remove(el); lst.insert(14, el)

# ---------------------------------------------------------------- conclusion
s = S[15]
set_text(shape(s, "TextBox 17").text_frame, [
    "Conclusion",
    ("3.08 M-parameter Swin-Mamba U-Net ", "segments lung nodules at 4.85 GMACs per 512² slice."),
    ("With 2.5D input ", "it beats U-Net (31 M) and Swin-UNet (41 M): raw test Dice 0.708, false alarms 12.1%."),
    ("Final inference pipeline: ", f"Dice {final.test_dice_global:.3f}, false alarms {final.test_fp_slice_rate * 100:.1f}%, "
                                   "with no change to model size.")], size=1700)

set_text(shape(s, "TextBox 18").text_frame, [
    "Future Scope: raising full-slice Dice",
    ("Two-stage cascade: ", "full-slice detector + crop refiner trained on real detections."),
    ("More 3D context: ", "5–7 input slices or a small 3D stage to reject vessels."),
    ("3D post-processing: ", "drop detections that appear on only one slice (fewer false alarms)."),
    ("More data: ", "all 1,018 LIDC-IDRI patients instead of 700."),
    ("Small-nodule training: ", "deep supervision, boundary loss, CBAM attention on skip connections."),
    ("5-fold cross-validation + fold ensemble: ", "reliable gains with error bars; external test on LUNA16.")], size=1500)

p.save(str(OUT))
print("saved", OUT, "| final step:", final.step, f"test dice {final.test_dice_global:.4f}")
