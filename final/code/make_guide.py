"""Complete team guide for the ROI (crop) version -> HTML (then Edge -> PDF).
Usage: python final/code/make_guide.py out.html"""
import base64, sys
from pathlib import Path

FIG = Path(r"D:\Claude\CV\crop_version\presentation\figures")
def img(name, w="100%", cap=""):
    b = base64.b64encode((FIG / name).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{b}" style="width:{w}"><figcaption>{cap}</figcaption></figure>'

def eq(s, n=""):
    return f'<div class="eq"><span>{s}</span><span class="en">{n}</span></div>'

def box(t, s, cls="note"):
    return f'<div class="{cls}"><b>{t}</b> {s}</div>'

H = []
A = H.append

# ------------------------------------------------------------------ cover
A("""<div class="cover">
<div class="kicker">Team study guide · ROI (crop) version</div>
<h1>Lightweight Swin-Mamba U-Net<br>for Lung Nodule Segmentation</h1>
<p class="sub">Everything about the project in one place: the medical background, the data pipeline, every layer of the model with its mathematics, training, evaluation, results, and a subject-wise glossary.</p>
<table class="kpi"><tr>
<td><b>3.08 M</b><span>parameters</span></td><td><b>0.818</b><span>test Dice (128×128 ROI, 2.5D)</span></td>
<td><b>0.30</b><span>GMACs per ROI (U-Net: 12.0)</span></td><td><b>700</b><span>LIDC-IDRI patients</span></td></tr></table>
<p class="toc-h">Contents</p>
<ol class="toc">
<li>The project in one page</li><li>Medical background</li><li>Dataset and preprocessing</li>
<li>Deep-learning foundations</li><li>Architecture overview and tensor shapes</li><li>Convolutional stem</li>
<li>Swin Transformer stages (attention maths)</li><li>Mamba / VSS stages (state-space maths)</li>
<li>Patch merging and the decoder</li><li>Where the 3.08 M parameters are</li><li>Loss function</li>
<li>Training procedure</li><li>Evaluation metrics</li><li>Baselines: U-Net and Swin-UNet</li>
<li>Results</li><li>Computational cost and "lightweight"</li><li>Limitations and future scope (full-slice segmentation)</li>
<li>Glossary by subject</li><li>Likely viva questions</li></ol>
</div>""")

# ------------------------------------------------------------------ 1
A("""<h2>1. The project in one page</h2>
<p><b>Goal.</b> A nodule has already been located (by a radiologist or a detector). Given a <b>128×128 region of interest (ROI)</b> cut from the CT slice around it, produce a <b>mask</b> of the same size in which every pixel is labelled 1 (nodule) or 0 (not nodule). This task is called <b>semantic segmentation</b>; here it means <i>outlining</i> a known nodule, the setting used by most published LIDC-IDRI segmentation papers.</p>
<p><b>Why it matters.</b> Lung cancer is the leading cause of cancer death. Found early as a small <b>nodule</b>, it is far more treatable. Outlining nodules by hand is slow and radiologists disagree, so an automatic, consistent tool helps measure size and growth over time.</p>
<p><b>What is new about ours.</b> Most strong segmentation networks have 30–150 million parameters. We designed a <b>hybrid</b> network of only <b>3.08 M parameters</b> that combines three building blocks, each used where it is strongest:</p>
<ul><li><b>Convolutions</b> (stem and decoder): sharp local detail such as edges.</li>
<li><b>Swin Transformer attention blocks</b> (high-resolution stages, red dashed box in Fig. 2): self-attention inside small 8×8 windows, for texture.</li>
<li><b>Mamba (VSS) blocks</b> (low-resolution stages): scan the whole image with cost that grows only <i>linearly</i> with image size, for global context.</li></ul>
<p>We also feed <b>2.5D input</b>: the slice plus its neighbours about 2 mm above and below, as three channels. This lets the network tell round nodules, which disappear within a few slices, from tube-like vessels, which continue.</p>
<p><b>How we judged it.</b> We trained two well-known models, U-Net (31 M) and Swin-UNet (41 M, ImageNet pre-trained), on exactly the same patches and settings. All models are tested on <b>1,379 nodule-centred 128×128 patches from 105 patients never seen in training</b>.</p>""")
A(img("fig_pipeline.png", "100%", "Fig. 1 — End-to-end pipeline."))
A(box("Result in one line:", "our 2.5D model reaches test Dice 0.818, only 0.010 below U-Net and 0.036 below pre-trained Swin-UNet, while being 10–13× smaller and using about 40× less computation."))

# ------------------------------------------------------------------ 2
A("""<h2>2. Medical background</h2>
<h3>2.1 CT scan</h3>
<p>A <b>computed tomography (CT)</b> scanner rotates an X-ray source around the patient and reconstructs a 3D volume. The volume is stored as a stack of 2D <b>axial slices</b> (horizontal cross-sections), each typically 512×512 pixels. Each pixel is about 0.5–0.9 mm, and slices are 0.6–3 mm apart (the <b>slice thickness / spacing</b>). The files are in <b>DICOM</b> format.</p>
<h3>2.2 Hounsfield Units (HU)</h3>
<p>Each CT pixel stores a density value in <b>Hounsfield Units</b>, defined relative to water and air:</p>""")
A(eq("HU = 1000 × (μ − μ<sub>water</sub>) / (μ<sub>water</sub> − μ<sub>air</sub>)", "μ = X-ray attenuation coefficient"))
A("""<table class="t"><tr class="h"><td>Tissue</td><td>Approx. HU</td></tr>
<tr><td>Air</td><td>−1000</td></tr><tr><td>Lung tissue</td><td>−900 to −500</td></tr><tr><td>Fat</td><td>−100</td></tr>
<tr><td>Water</td><td>0</td></tr><tr><td>Soft tissue / nodule / blood</td><td>+20 to +80</td></tr><tr><td>Bone</td><td>+400 to +1000+</td></tr></table>
<h3>2.3 Lung window</h3>
<p>A screen shows only 256 grey levels, but HU spans about 4000 values. <b>Windowing</b> clips HU to a range of interest and stretches that range to 0–255. We use the lung window <b>[−1000, 400] HU</b>:</p>""")
A(eq("I = 255 × (clip(HU, −1000, 400) + 1000) / 1400"))
A("""<h3>2.4 Lung nodule</h3>
<p>A <b>pulmonary nodule</b> is a small, roughly round lesion of 3–30 mm. Most are benign, but some are early cancer. Hard cases: tiny nodules of a few pixels, nodules touching the chest wall (<b>juxta-pleural</b>), faint <b>ground-glass</b> nodules, and nodules next to vessels, which look identical in a single slice.</p>
<h3>2.5 LIDC-IDRI dataset</h3>
<p>The <b>Lung Image Database Consortium and Image Database Resource Initiative</b> is a public set of 1,018 chest CT scans. Up to <b>4 radiologists</b> independently outlined each nodule of 3 mm or larger. Because they disagree, we use a <b>50% consensus</b> mask: a pixel is "nodule" if at least half of the radiologists who marked that nodule included it. We used <b>700 patients</b>.</p>""")

# ------------------------------------------------------------------ 3
A("""<h2>3. Dataset and preprocessing</h2>
<h3>3.1 Steps (<code>preprocess_lidc.py</code>)</h3>
<ol><li>Read each scan with <b>pylidc</b> and convert it to a 3D HU volume (512×512×Z).</li>
<li>Build the nodule mask: for each nodule, take the 50% consensus of its annotations, then combine (logical OR) all nodules.</li>
<li><b>Choose slices.</b> For the ROI protocol only slices that <b>contain a nodule</b> are used (the full-slice prototype in §17 also used nodule-free slices).</li>
<li>Apply the lung window and save the image and mask as 512×512 PNG files.</li>
<li><b>Patient-wise split</b> 70/15/15. All slices of one patient go to exactly one split, so the test set contains <i>people the model has never seen</i>. Splitting by slice would leak near-identical neighbouring slices into the test set and inflate scores.</li></ol>
<p><b>Cutting ROI patches</b> (<code>crop_version/code/train_crop.py</code>): each connected nodule region on a slice is one nodule. Validation and test: one fixed 128×128 patch centred on each nodule's centroid (shifted inwards at the image border). Training: on each slice pick one nodule at random and move the patch centre by up to ±24 px, so the nodule is not always exactly in the middle.</p>
<table class="t"><tr class="h"><td>Split</td><td>Patients</td><td>Nodule slices</td><td>128×128 patches</td></tr>
<tr><td>Train</td><td>490</td><td>6,101</td><td>one new random patch per slice each epoch</td></tr>
<tr><td>Validation</td><td>105</td><td>1,763</td><td>1,960 (one per nodule region)</td></tr>
<tr><td>Test</td><td>105</td><td>1,274</td><td>1,379 (one per nodule region)</td></tr></table>
<h3>3.2 2.5D input (<code>make_25d.py</code>)</h3>
<p>For every kept slice <i>z</i> we also take slices <i>z−k</i> and <i>z+k</i>, where <b>k = max(1, round(2 mm / slice spacing))</b>, so the neighbours are about 2 mm away whatever the scanner. The three slices are stacked as the 3 channels of one image (like RGB). The mask stays the mask of the middle slice.</p>""")
A(box("Why it is called 2.5D:", "2D networks see one slice and 3D networks see a whole volume. We feed a thin slab of 3 slices into a 2D network, so it gets some depth information at almost exactly 2D cost (only +288 parameters, all in the first convolution).", "note"))
A("""<h3>3.3 Data augmentation (training only)</h3>
<p>We generate random variations so the network does not memorise the training images:</p>
<ul><li><b>Random jitter:</b> the 128×128 patch centre is moved by up to ±24 px around the nodule, so the model does not learn that "the nodule is always in the centre".</li>
<li>Horizontal and vertical flips; rotations by 0°, 90°, 180° or 270°.</li>
<li>Brightness and contrast jitter: I′ = a·I + b, with a ∈ [0.9, 1.1] and b ∈ [−0.05, 0.05].</li>
<li>Gaussian noise with σ = 0.02 (30% of the time).</li>
<li>2.5D only: swap the "above" and "below" channels, since the scan direction should not matter.</li></ul>
<h3>3.4 Normalisation</h3>""")
A(eq("x = (I/255 − 0.5) / 0.25", "centres pixel values near 0 with spread about 1"))

# ------------------------------------------------------------------ 4
A("""<h2>4. Deep-learning foundations</h2>
<h3>4.1 Parameters</h3>
<p>A neural network is a long chain of simple maths operations containing adjustable numbers called <b>parameters</b> (weights and biases). Training tunes them so the output matches the radiologists' masks. "3.08 M parameters" means 3,081,517 such numbers. Fewer parameters means a smaller file (about 12 MB), less memory, and usually faster inference.</p>
<h3>4.2 Convolution</h3>
<p>A small k×k filter slides over the image. At each position it multiplies and sums:</p>""")
A(eq("y[c<sub>o</sub>, i, j] = b[c<sub>o</sub>] + Σ<sub>c<sub>i</sub></sub> Σ<sub>u,v</sub> W[c<sub>o</sub>, c<sub>i</sub>, u, v] · x[c<sub>i</sub>, i+u, j+v]",
      "params = C<sub>in</sub>·C<sub>out</sub>·k² (+C<sub>out</sub>)"))
A("""<p><b>Stride 2</b> moves the filter 2 pixels at a time, halving height and width. <b>Padding</b> adds zeros at the border to keep the size. A <b>depthwise</b> convolution filters each channel separately, which is very cheap. A <b>1×1 convolution</b> mixes channels at each pixel.</p>
<h3>4.3 Activation functions</h3>""")
A(eq("GELU(x) = x · Φ(x)", "Φ = standard normal CDF; smooth version of ReLU = max(0,x)"))
A(eq("SiLU(x) = x · σ(x),&nbsp;&nbsp; σ(x) = 1/(1+e<sup>−x</sup>)", "σ = sigmoid, squashes to (0,1)"))
A(eq("softplus(x) = log(1 + e<sup>x</sup>)", "always positive; used for Mamba's step size Δ"))
A("""<h3>4.4 Normalisation layers</h3>""")
A(eq("BatchNorm: y = γ·(x − μ<sub>batch</sub>)/√(σ²<sub>batch</sub>+ε) + β", "statistics per channel over the batch (used in conv layers)"))
A(eq("LayerNorm: y = γ·(x − μ<sub>token</sub>)/√(σ²<sub>token</sub>+ε) + β", "statistics over the channels of each pixel/token (used in Swin/Mamba)"))
A("""<h3>4.5 Other building blocks</h3>
<ul><li><b>Linear (fully connected) layer:</b> y = Wx + b. With C<sub>in</sub> inputs and C<sub>out</sub> outputs it has C<sub>in</sub>·C<sub>out</sub> + C<sub>out</sub> parameters.</li>
<li><b>MLP:</b> Linear → GELU → Linear, applied to each token separately (hidden size = ratio × C).</li>
<li><b>Residual connection:</b> y = x + F(x). The block only learns a correction. This keeps gradients flowing in deep networks.</li>
<li><b>DropPath (stochastic depth):</b> during training, randomly skip a whole residual branch with probability p (0 → 0.1, rising linearly with depth). This is a form of regularisation.</li>
<li><b>Token:</b> in transformer and Mamba stages, each pixel of the feature map is treated as a vector of C numbers, called a token.</li>
<li><b>Feature map:</b> the intermediate "image" inside the network, of shape channels × height × width.</li>
<li><b>Encoder–decoder (U-Net) idea:</b> the encoder shrinks the image while increasing channels (what is it?). The decoder enlarges it back (where exactly?). <b>Skip connections</b> pass fine detail from encoder to decoder.</li></ul>""")

# ------------------------------------------------------------------ 5
A("""<h2>5. Architecture overview and tensor shapes</h2>""")
A(img("fig_swinmamba.png", "100%", "Fig. 2 — Swin-Mamba U-Net. Two Swin stages at high resolution, two Mamba (VSS) stages at low resolution, conv decoder with skip connections."))
A("""<p>The table traces one 2.5D input through the network (B = batch size; dims = 32, 64, 128, 256; 2 blocks per stage). The input size must be divisible by 32; for the ROI version it is 128×128.</p>
<table class="t"><tr class="h"><td>Step</td><td>Operation</td><td>Output shape (C × H × W)</td><td>Role</td></tr>
<tr><td>Input</td><td>3 slices</td><td>3 × 128 × 128</td><td>z−k, z, z+k</td></tr>
<tr><td>Stem 1</td><td>Conv3×3 s2 + BN + GELU</td><td>16 × 64 × 64</td><td>skip s0</td></tr>
<tr><td>Stem 2</td><td>Conv3×3 s2 + BN + GELU + Conv1×1</td><td>32 × 32 × 32</td><td>patch embedding</td></tr>
<tr><td>Stage 1</td><td>2 × Swin block (W-MSA, SW-MSA), 2 heads</td><td>32 × 32 × 32</td><td>skip f1 · local texture</td></tr>
<tr><td>Merge</td><td>Patch merging</td><td>64 × 16 × 16</td><td></td></tr>
<tr><td>Stage 2</td><td>2 × Swin block, 4 heads</td><td>64 × 16 × 16</td><td>skip f2</td></tr>
<tr><td>Merge</td><td>Patch merging</td><td>128 × 8 × 8</td><td></td></tr>
<tr><td>Stage 3</td><td>2 × VSS (Mamba) block</td><td>128 × 8 × 8</td><td>skip f3 · global context</td></tr>
<tr><td>Merge</td><td>Patch merging</td><td>256 × 4 × 4</td><td></td></tr>
<tr><td>Stage 4</td><td>2 × VSS block (bottleneck)</td><td>256 × 4 × 4</td><td>f4</td></tr>
<tr><td>Up 3</td><td>Upsample + concat f3 + 2 conv</td><td>128 × 8 × 8</td><td></td></tr>
<tr><td>Up 2</td><td>Upsample + concat f2 + 2 conv</td><td>64 × 16 × 16</td><td></td></tr>
<tr><td>Up 1</td><td>Upsample + concat f1 + 2 conv</td><td>32 × 32 × 32</td><td></td></tr>
<tr><td>Up 0</td><td>Upsample + concat s0 + 2 conv</td><td>16 × 64 × 64</td><td></td></tr>
<tr><td>Resize</td><td>Bilinear ×2</td><td>16 × 128 × 128</td><td></td></tr>
<tr><td>Head</td><td>Conv3×3 + BN + GELU + Conv1×1</td><td>1 × 128 × 128</td><td>logit per pixel</td></tr>
<tr><td>Output</td><td>sigmoid, threshold 0.5</td><td>1 × 128 × 128</td><td>binary mask</td></tr></table>""")
A(box("Design logic:", "attention cost grows with the number of tokens, so Swin's <i>local windows</i> are used where the map is large (32² and 16² for a 128² ROI). Mamba's <i>linear-cost global scan</i> is used at the coarse stages (8² and 4²), where one token covers 16–32 pixels of the original image. The same network runs at any input size divisible by 32, including full 512² slices."))

# ------------------------------------------------------------------ 6
A("""<h2>6. Convolutional stem</h2>
<p>Two stride-2 3×3 convolutions reduce 128 → 64 → 32 and expand channels 3 → 16 → 32. Convolutions are good at low-level edges and much cheaper than attention at full resolution. The 16×64×64 output (s0) is kept as the highest-resolution skip connection so the decoder can recover sharp nodule borders.</p>
<p>Parameters: stem1 = 3·16·9 (conv) + 32 (BN) = <b>464</b>. Stem2 = 16·32·9 + 64 + (32·32+32) = <b>5,728</b>. The only difference between the 2D and 2.5D models is in stem1: 2 extra input channels × 16 × 9 = <b>288</b> parameters.</p>""")

# ------------------------------------------------------------------ 7
A("""<h2>7. Swin Transformer stages</h2>
<h3>7.1 Self-attention</h3>
<p>Each token decides how much to "look at" every other token. From the input X (N tokens × C) we compute <b>queries</b>, <b>keys</b> and <b>values</b>:</p>""")
A(eq("Q = XW<sub>Q</sub>,&nbsp; K = XW<sub>K</sub>,&nbsp; V = XW<sub>V</sub>"))
A(eq("Attention(Q,K,V) = softmax( QK<sup>T</sup> / √d + B ) V", "d = channels per head, B = relative position bias"))
A("""<p>QK<sup>T</sup> is an N×N table of similarity scores. Softmax turns each row into weights summing to 1, and the output for each token is a weighted average of the values. Dividing by √d keeps the scores from growing too large. <b>Multi-head</b> attention splits C channels into h heads (2, 4, 8, 16 per stage), each attending independently, and then concatenates them.</p>
<h3>7.2 Why windows</h3>
<p>Global attention over N tokens costs O(N²). On a full slice the stage-1 map is 128×128, N = 16,384 tokens, giving 268 million pairs per head per image, which is far too much. Swin splits the map into non-overlapping <b>8×8 windows</b> (64 tokens) and runs attention only inside each window:</p>""")
A(eq("Cost<sub>global</sub> ∝ N² C &nbsp;&nbsp;→&nbsp;&nbsp; Cost<sub>window</sub> ∝ N · M² · C", "M = 8 window size; linear in N"))
A("""<p>On a 128² ROI the stage-1 map is 32×32: 16 windows × 64 tokens. Each window computes a 64×64 attention table.</p>
<h3>7.3 Shifted windows</h3>
<p>Windows alone never talk to each other. So every <b>second</b> Swin block cyclically shifts the feature map by M/2 = 4 pixels (torch.roll) before partitioning. The new windows straddle the old borders, so information crosses windows. A <b>mask</b> (value −100 before the softmax, so the weight is about 0) stops tokens that were wrapped around from the opposite edge from attending to each other. Afterwards the map is shifted back.</p>
<h3>7.4 Relative position bias</h3>
<p>Attention by itself does not know where tokens are. Swin adds a learnable bias B<sub>ij</sub> that depends only on the <i>relative</i> offset (Δy, Δx) between tokens i and j. There are (2M−1)² = 225 possible offsets per head, stored in a small table.</p>
<h3>7.5 One Swin block</h3>""")
A(eq("X′ = X + DropPath( (S)W-MSA( LN(X) ) )"))
A(eq("Y = X′ + DropPath( MLP( LN(X′) ) )", "MLP hidden = 4C"))
A("""<p>Parameters per block with C channels: QKV 3C²+3C, projection C²+C, MLP 8C²+5C, two LayerNorms 4C, bias table 225·heads. For C = 32: about 12.9 K per block, and stage 1 (2 blocks) = <b>26,308</b>. Stage 2 (C = 64) = <b>101,768</b>.</p>""")
A(img("fig_blocks.png", "92%", "Fig. 3 — Swin block and VSS (Mamba) block internals."))

# ------------------------------------------------------------------ 8
A("""<h2>8. Mamba / VSS stages: the state-space maths</h2>
<h3>8.1 Intuition</h3>
<p>Mamba reads the image as a <b>sequence</b> of tokens, like reading a page line by line. It keeps a small running memory (the <b>hidden state</b> h) that summarises everything seen so far. At each token it decides how much to forget and how much new information to store. One pass touches each token once, so the cost is <b>linear</b> in N, compared with quadratic for global attention. Even so, the last token's output depends on the whole image.</p>
<h3>8.2 Continuous state-space model (SSM)</h3>
<p>The classical linear system from control theory, for an input signal u(t), state h(t) ∈ ℝ<sup>N</sup> and output y(t):</p>""")
A(eq("h′(t) = A h(t) + B u(t),&nbsp;&nbsp;&nbsp; y(t) = C h(t) + D u(t)", "A: N×N, B: N×1, C: 1×N, D: scalar skip"))
A("""<h3>8.3 Discretisation (zero-order hold)</h3>
<p>Computers work in steps. With a step size Δ > 0, the continuous system becomes a recurrence:</p>""")
A(eq("Ā = exp(Δ A),&nbsp;&nbsp; B̄ ≈ Δ B"))
A(eq("h<sub>t</sub> = Ā h<sub>t−1</sub> + B̄ u<sub>t</sub>,&nbsp;&nbsp;&nbsp; y<sub>t</sub> = C h<sub>t</sub> + D u<sub>t</sub>"))
A("""<p>We use a <b>diagonal</b> A with negative entries, A = −exp(A_log), initialised to −1, −2, …, −N (N = d_state = 8). Then Ā = exp(ΔA) lies between 0 and 1 and acts as a <b>forget gate</b>. A large Δ makes Ā small (forget the past, focus on the current token). A small Δ makes Ā ≈ 1 (keep remembering).</p>
<h3>8.4 "Selective": what makes it Mamba</h3>
<p>In older SSMs, Δ, B and C were fixed. In Mamba they are <b>computed from the input at every token</b>, so the model chooses, based on content, what to remember. For example it can keep "I am inside a round bright blob" and forget a passing vessel:</p>""")
A(eq("Δ<sub>t</sub> = softplus( W<sub>Δ</sub> (W<sub>r</sub> u<sub>t</sub>) + b<sub>Δ</sub> ),&nbsp;&nbsp; B<sub>t</sub> = W<sub>B</sub> u<sub>t</sub>,&nbsp;&nbsp; C<sub>t</sub> = W<sub>C</sub> u<sub>t</sub>",
      "low-rank R = ⌈C/16⌉ for Δ"))
A("""<h3>8.5 2D selective scan (SS2D): four directions</h3>
<p>An image is 2D, but a scan is 1D. The VSS block flattens the H×W map into a sequence of L = H·W tokens in <b>four orders</b>: row by row, column by column, and both reversed. It runs an independent selective scan on each and sums the four outputs back on the grid. Every pixel therefore receives context from all four directions.</p>
<h3>8.6 One VSS block (our implementation)</h3>
<ol><li>LayerNorm, then a Linear layer expands C → 2·D (D = 2C), split into x and gate z.</li>
<li>x passes through a <b>depthwise 3×3 convolution</b> and SiLU, which gives local mixing before the scan.</li>
<li>4-direction flatten → per direction compute Δ, B, C → selective scan → merge.</li>
<li>LayerNorm(y) × SiLU(z) (gating) → Linear D → C → residual add.</li>
<li>LayerNorm → MLP (hidden 2C) → residual add.</li></ol>""")
A(eq("X′ = X + DropPath( SS2D( LN(X) ) ),&nbsp;&nbsp;&nbsp; Y = X′ + DropPath( MLP( LN(X′) ) )"))
A("""<h3>8.7 Our pure-PyTorch chunked scan (an engineering contribution)</h3>
<p>The official Mamba uses a custom CUDA kernel that does not install on Windows. We wrote the scan in plain PyTorch. The sequence is cut into chunks of 8 tokens. Inside a chunk the recurrence is solved in parallel with cumulative sums in log space:</p>""")
A(eq("h<sub>t</sub> = e<sup>S<sub>t</sub></sup> · Σ<sub>s≤t</sub> e<sup>−S<sub>s</sub></sup> B̄<sub>s</sub>u<sub>s</sub> + e<sup>S<sub>t</sub></sup> h<sub>start</sub>,&nbsp;&nbsp; S<sub>t</sub> = Σ<sub>s≤t</sub> Δ<sub>s</sub>A",
      "then chunks are linked by a short loop"))
A(box("Numerical-stability fix:", "e<sup>−S</sup> can overflow float32 (in our first attempt, chunk 16 reached e<sup>80</sup> and gave NaN). We clamp each step's log-decay Δ·A ≥ −4 and use chunk 8, so |S| ≤ 32 and e<sup>±32</sup> is safe. The scan always runs in float32, even under bf16 mixed precision.", "warn"))
A("""<p>Parameters: stage 3 (C = 128) = <b>421,632</b>; stage 4 (C = 256) = <b>1,564,160</b>, the largest single part because parameters grow with C².</p>""")

# ------------------------------------------------------------------ 9
A("""<h2>9. Patch merging and the decoder</h2>
<h3>9.1 Patch merging (downsampling)</h3>
<p>Take each 2×2 group of tokens, concatenate them (4C channels), apply LayerNorm, then a Linear layer 4C → 2C. Height and width halve, channels double. Parameters = 4C·2C + 8C (norm): <b>8,448</b> (C = 32), <b>33,280</b> (64), <b>132,096</b> (128).</p>
<h3>9.2 Decoder (UpBlock)</h3>
<p>Each UpBlock: bilinear upsample to the skip's size → concatenate with the skip feature → two (Conv3×3 + BN + GELU). Bilinear interpolation computes each new pixel as a distance-weighted average of the 4 nearest old pixels, and has no parameters.</p>
<p>Up3 = <b>590,336</b> params (input 256+128 channels), Up2 = 147,712, Up1 = 36,992, Up0 = 9,280. A final head (Conv3×3, then Conv1×1 to 1 channel) = 2,353 produces one <b>logit</b> z per pixel. The probability is p = σ(z), and the mask is p > 0.5.</p>
<h3>9.3 Why skip connections</h3>
<p>Deep features know <i>what</i> is there but have lost the exact <i>where</i>: at 16×16, one token covers 32×32 pixels. The skips bring back the precise edges from the earlier, high-resolution layers. This matters for nodules that are only 5–20 pixels wide.</p>""")

# ------------------------------------------------------------------ 10
A("""<h2>10. Where the 3.08 M parameters are (2.5D model)</h2>
<table class="t"><tr class="h"><td>Part</td><td>Parameters</td><td>Share</td></tr>
<tr><td>Stem (2 convs)</td><td>6,192</td><td>0.2%</td></tr>
<tr><td>Stage 1 · Swin, C=32</td><td>26,308</td><td>0.9%</td></tr>
<tr><td>Stage 2 · Swin, C=64</td><td>101,768</td><td>3.3%</td></tr>
<tr><td>Stage 3 · Mamba, C=128</td><td>421,632</td><td>13.7%</td></tr>
<tr><td>Stage 4 · Mamba, C=256</td><td>1,564,160</td><td>50.8%</td></tr>
<tr><td>Patch merging ×3 + stage norms</td><td>174,784</td><td>5.7%</td></tr>
<tr><td>Decoder (4 UpBlocks)</td><td>784,320</td><td>25.5%</td></tr>
<tr><td>Head</td><td>2,353</td><td>0.1%</td></tr>
<tr><td><b>Total</b></td><td><b>3,081,517</b></td><td>100%</td></tr></table>
<p>2D version: 3,081,229 (288 fewer). For comparison, U-Net has 31,036,481 and Swin-UNet 41,389,765.</p>""")

# ------------------------------------------------------------------ 11
A("""<h2>11. Loss function</h2>
<p>The <b>loss</b> is a single number measuring how wrong the prediction is. Training lowers it. With p<sub>i</sub> = σ(z<sub>i</sub>) as the predicted probability and g<sub>i</sub> ∈ {0,1} as the ground truth for pixel i:</p>""")
A(eq("L<sub>BCE</sub> = −(1/n) Σ<sub>i</sub> [ g<sub>i</sub> log p<sub>i</sub> + (1−g<sub>i</sub>) log(1−p<sub>i</sub>) ]", "binary cross-entropy: per-pixel correctness"))
A(eq("L<sub>Dice</sub> = 1 − (2 Σ p<sub>i</sub>g<sub>i</sub> + 1) / (Σ p<sub>i</sub> + Σ g<sub>i</sub> + 1)", "soft Dice: overlap, +1 smoothing"))
A(eq("L = L<sub>BCE</sub> + L<sub>Dice</sub>"))
A(box("Why both:", "a nodule is often under 0.1% of the pixels. BCE alone can score 99.9% by predicting \"no nodule everywhere\". Dice measures overlap with the small object and ignores the huge background, but it is unstable on its own. Together they are stable and focus on the nodule."))

# ------------------------------------------------------------------ 12
A("""<h2>12. Training procedure</h2>
<table class="t"><tr class="h"><td>Setting</td><td>Value</td><td>Meaning</td></tr>
<tr><td>Optimiser</td><td>AdamW</td><td>Adam with decoupled weight decay</td></tr>
<tr><td>Learning rate</td><td>5×10<sup>−4</sup> (Swin-UNet 2×10<sup>−4</sup>)</td><td>step size of updates</td></tr>
<tr><td>Weight decay</td><td>0.05 (not on norms, biases, A_log, D, dt_bias)</td><td>pulls weights towards 0 to reduce overfitting</td></tr>
<tr><td>Schedule</td><td>2-epoch linear warm-up + cosine decay</td><td>start gently, then slowly lower the LR</td></tr>
<tr><td>Epochs</td><td>60</td><td>full passes over the training set</td></tr>
<tr><td>Batch size</td><td>32 patches of 128×128 (Swin-UNet 16, GPU memory)</td><td>images per update step</td></tr>
<tr><td>Precision</td><td>bfloat16 mixed (scan in fp32)</td><td>faster, less memory</td></tr>
<tr><td>Gradient clipping</td><td>‖g‖ ≤ 1.0</td><td>prevents exploding updates</td></tr>
<tr><td>Safety</td><td>skip a batch if loss or grad is NaN/inf</td><td>keeps weights intact</td></tr>
<tr><td>Model selection</td><td>best validation Dice → best.pt</td><td>the test set is never used for choices</td></tr>
<tr><td>Hardware</td><td>RTX 4050 Laptop, 6 GB</td><td>about 45 min – 2.2 h per model</td></tr></table>
<h3>12.1 Gradient descent and backpropagation</h3>
<p><b>Backpropagation</b> uses the chain rule to compute ∂L/∂θ, how the loss changes with each parameter θ. The optimiser then moves θ a small step against that gradient.</p>
<h3>12.2 AdamW update</h3>""")
A(eq("m<sub>t</sub> = β<sub>1</sub>m<sub>t−1</sub> + (1−β<sub>1</sub>) g<sub>t</sub>,&nbsp;&nbsp; v<sub>t</sub> = β<sub>2</sub>v<sub>t−1</sub> + (1−β<sub>2</sub>) g<sub>t</sub>²", "running mean and variance of gradients (β₁=0.9, β₂=0.999)"))
A(eq("θ<sub>t</sub> = θ<sub>t−1</sub> − η<sub>t</sub> [ m̂<sub>t</sub> / (√v̂<sub>t</sub> + ε) + λ θ<sub>t−1</sub> ]", "m̂, v̂ bias-corrected; λ = weight decay"))
A("""<h3>12.3 Learning-rate schedule</h3>""")
A(eq("η(s) = η<sub>0</sub>·(s+1)/S<sub>w</sub> &nbsp;(s &lt; S<sub>w</sub>);&nbsp;&nbsp; η(s) = η<sub>0</sub>·½[1 + cos(π (s−S<sub>w</sub>)/(S−S<sub>w</sub>))] &nbsp;(otherwise)",
      "S<sub>w</sub> = warm-up steps, S = total steps"))
A(img("fig_curves.png", "100%", "Fig. 4 — Validation Dice and training loss per epoch for all four models (ROI protocol)."))
A("""<table class="t"><tr class="h"><td>Model</td><td>Best val Dice</td><td>At epoch</td><td>Train time</td></tr>
<tr><td>Swin-Mamba 2.5D</td><td>0.838</td><td>44</td><td>48 min</td></tr>
<tr><td>Swin-Mamba 2D</td><td>0.830</td><td>31</td><td>48 min</td></tr>
<tr><td>U-Net 2D</td><td>0.845</td><td>38</td><td>46 min</td></tr>
<tr><td>Swin-UNet 2D (ImageNet)</td><td>0.859</td><td>53</td><td>2.2 h</td></tr></table>
<p>Why so fast compared with full slices (6–12 h)? A 128² patch has 16× fewer pixels than a 512² slice, only nodule slices are used (6,101 instead of 12,268), and batches are larger. The task is also easier: the model only has to outline a nodule it is shown.</p>""")

# ------------------------------------------------------------------ 13
A("""<h2>13. Evaluation metrics</h2>
<p>Each test pixel is one of: <b>TP</b> (true positive: predicted nodule, is nodule), <b>FP</b> (false positive: predicted nodule, is not), <b>FN</b> (false negative: missed nodule), or <b>TN</b> (true negative: correct background).</p>""")
A(eq("Dice = 2TP / (2TP + FP + FN) = 2|P∩G| / (|P| + |G|)", "0 = no overlap, 1 = perfect"))
A(eq("IoU (Jaccard) = TP / (TP + FP + FN),&nbsp;&nbsp; Dice = 2·IoU / (1 + IoU)"))
A(eq("Precision = TP / (TP + FP),&nbsp;&nbsp; Recall (sensitivity) = TP / (TP + FN)"))
A("""<ul><li><b>Global Dice</b> (our main number): TP, FP and FN summed over <i>all</i> pixels of <i>all</i> 1,379 test patches, then one Dice computed. Large nodules weigh more.</li>
<li><b>Dice per patch:</b> Dice computed on each patch, then averaged. Every nodule counts equally, so small ones pull it down.</li>
<li><b>False alarms</b> are <i>not</i> measured in the ROI protocol, because every test patch contains a nodule. The full-slice prototype (§17) measures them.</li>
<li><b>Accuracy</b> is not used. Since 99.9% of pixels are background, an all-black mask would score 99.9% accuracy.</li></ul>""")

# ------------------------------------------------------------------ 14
A("""<h2>14. Baselines</h2>""")
A(img("fig_baselines.png", "95%", "Fig. 5 — The two comparison models."))
A("""<p><b>U-Net</b> (Ronneberger 2015): the classic medical segmentation CNN. 4 down-sampling levels of double 3×3 convolutions (64 → 1024 channels), mirrored decoder with skips. Pure convolution, so it sees only local context unless very deep. <b>31.04 M</b> parameters, trained from scratch.</p>
<p><b>Swin-UNet</b> (Cao 2022): a U-shaped network made entirely of Swin Transformer blocks, encoder initialised with <b>ImageNet-pretrained</b> Swin-Tiny weights. <b>41.39 M</b> parameters.</p>
<p><b>Fair comparison:</b> same patches, split, augmentation, loss, epochs, schedule and test protocol for all. Only the network changes. Exceptions forced by hardware or design: Swin-UNet used batch 16 and learning rate 2×10<sup>−4</sup>, and internally upsamples a 128² patch ×2 because its 4×4 bottleneck would be smaller than its 8×8 window.</p>""")

# ------------------------------------------------------------------ 15
A("""<h2>15. Results (test set: 105 patients, 1,379 ROI patches of 128×128)</h2>
<table class="t"><tr class="h"><td>Model</td><td>Params</td><td>GMACs</td><td>Dice</td><td>Dice (per patch)</td><td>IoU</td><td>Precision</td><td>Recall</td></tr>
<tr class="hl"><td><b>Swin-Mamba 2.5D (ours)</b></td><td><b>3.08 M</b></td><td><b>0.30</b></td><td>0.818</td><td>0.724</td><td>0.692</td><td>0.878</td><td>0.766</td></tr>
<tr><td>Swin-Mamba 2D (ours)</td><td>3.08 M</td><td>0.30</td><td>0.788</td><td>0.676</td><td>0.650</td><td>0.867</td><td>0.722</td></tr>
<tr><td>U-Net 2D</td><td>31.04 M</td><td>12.0</td><td>0.828</td><td>0.729</td><td>0.707</td><td>0.876</td><td>0.785</td></tr>
<tr><td>Swin-UNet 2D (ImageNet)</td><td>41.39 M</td><td>11.3</td><td><b>0.854</b></td><td><b>0.764</b></td><td><b>0.745</b></td><td><b>0.892</b></td><td><b>0.818</b></td></tr></table>
<h3>How to read it</h3>
<ul><li><b>Pre-trained Swin-UNet is the most accurate on crops</b> (0.854). A 128² patch centred on one object looks like a natural-image crop, so ImageNet features transfer well. Say this openly.</li>
<li><b>Ours 2.5D is only 0.010 Dice below U-Net</b> with 10× fewer parameters and 40× less computation, and has the same precision.</li>
<li><b>2.5D beats 2D by +0.030 Dice</b> (+0.048 per patch) at no extra cost: neighbouring slices separate round nodules from tube-like vessels.</li>
<li>Precision > recall for every model: predicted outlines are usually correct but slightly too small, or miss faint edges.</li></ul>""")
A(img("fig_qualitative.png", "82%", "Fig. 6 — Test patches ordered by nodule size (not hand-picked). Green = ground truth, red = prediction, per-model Dice in each panel."))
A(box("Is the crop version legitimate?", "Yes, as long as it is described correctly. It is the standard protocol of most LIDC-IDRI segmentation papers (S3TU-Net, for example, uses 128×128 nodule-centred patches). It measures how well a <i>known</i> nodule is outlined, not whether the model can find it. Our split is <b>by patient</b> (stricter than the slice-wise 9:1 split some papers use), and validation was used for every choice. Training was fast simply because patches are 16× smaller than full slices.", "warn"))

# ------------------------------------------------------------------ 16
A("""<h2>16. Computational cost and "lightweight"</h2>
<p><b>MAC</b> = one multiply-accumulate (a×b + c). <b>GMACs</b> = billions of MACs to process one image. FLOPs ≈ 2 × MACs.</p>
<table class="t"><tr class="h"><td>Model</td><td>Params</td><td>GMACs @ 128² ROI</td><td>Model file</td></tr>
<tr class="hl"><td>Swin-Mamba (ours)</td><td>3.08 M</td><td>0.30</td><td>about 12 MB</td></tr>
<tr><td>U-Net</td><td>31.04 M</td><td>12.0</td><td>about 124 MB</td></tr>
<tr><td>Swin-UNet</td><td>41.39 M</td><td>11.3 (with internal ×2 upsampling)</td><td>about 166 MB</td></tr></table>""")
A(img("fig_tradeoff.png", "100%", "Fig. 7 — Accuracy versus size and computation."))
A(box("Honest note:", "we are lightweight in <b>parameters (10–13× fewer)</b> and <b>computation (about 40× fewer GMACs per ROI)</b>. Measured GPU speed is similar to the baselines, because our pure-PyTorch scan is overhead-bound, not arithmetic-bound. An optimised CUDA scan kernel, or CPU / edge deployment, would turn the lower GMACs into real speed.", "warn"))

# ------------------------------------------------------------------ 17
A("""<h2>17. Limitations and future scope</h2>
<h3>17.1 Limitations</h3>
<ul><li>The ROI protocol <b>assumes the nodule has already been found</b>; it does not test detection or false alarms on healthy tissue.</li>
<li>On crops, pre-trained Swin-UNet (0.854) and U-Net (0.828) are more accurate than our model (0.818); our advantage is size and computation.</li>
<li>Each model was trained once with one seed; baselines were trained in 2D only; one dataset (no external test set yet).</li>
<li>Radiologists themselves agree only about 0.85–0.88 Dice on LIDC, which caps any model.</li>
<li>The GPU speed advantage is not yet realised (see §16).</li></ul>
<h3>17.2 Future scope: full-slice segmentation (already prototyped)</h3>
<p>The natural next step is to drop the "nodule already found" assumption: give the model the <b>whole 512×512 slice</b> and let it find <i>and</i> outline the nodule. We already trained all four models this way on the same 700 patients (with an equal number of nodule-free slices) and tested them on <b>2,558 full slices of the 105 test patients, 1,284 of which have no nodule</b>. Any prediction of 10 or more pixels on an empty slice is a false alarm.</p>
<table class="t"><tr class="h"><td>Full 512×512 slices</td><td>Params</td><td>Dice</td><td>Precision</td><td>Recall</td><td>False alarms ↓</td></tr>
<tr><td><b>Swin-Mamba 2.5D (ours)</b></td><td>3.08 M</td><td><b>0.708</b></td><td>0.744</td><td><b>0.676</b></td><td><b>12.1%</b></td></tr>
<tr><td>Swin-Mamba 2D (ours)</td><td>3.08 M</td><td>0.669</td><td>0.727</td><td>0.620</td><td>16.0%</td></tr>
<tr><td>U-Net 2D</td><td>31.04 M</td><td>0.696</td><td>0.745</td><td>0.652</td><td>14.4%</td></tr>
<tr><td>Swin-UNet 2D</td><td>41.39 M</td><td>0.670</td><td>0.715</td><td>0.631</td><td>18.2%</td></tr>
<tr class="hl"><td><b>Ours 2.5D + flip-averaging (TTA) + small-spot removal</b></td><td>3.08 M</td><td><b>0.723</b></td><td><b>0.791</b></td><td>0.665</td><td><b>6.9%</b></td></tr></table>
<ul><li><b>On full slices our model ranks first</b> of the four networks, the opposite of the crop ranking. Finding the nodule among vessels is where 2.5D context and global Mamba context pay off.</li>
<li>Full-slice Dice (0.72) is lower than ROI Dice (0.82) because the task is harder, not because the model is worse.</li>
<li><b>Tried, not adopted:</b> fine-tuning with a focal Tversky loss (0.721), a 2D+2.5D ensemble (0.717) and a two-stage crop refiner (0.717). They improved validation but not the test patients.</li>
<li><b>Next steps:</b> a detector + ROI cascade trained on realistic candidate patches (combining both versions), more slices of 3D context, 2.5D and pre-trained baselines, multiple seeds, external validation (LUNA16 / LNDb), a fused CUDA scan and quantised edge deployment.</li></ul>""")

# ------------------------------------------------------------------ 18
def gl(title, rows):
    return f'<h3>{title}</h3><table class="t gl">' + "".join(f"<tr><td><b>{a}</b></td><td>{b}</td></tr>" for a, b in rows) + "</table>"
A("<h2>18. Glossary by subject</h2>")
A(gl("Medical imaging", [
    ("CT", "Computed tomography: X-ray-based 3D imaging, stored as slices"),
    ("Axial slice", "Horizontal cross-section image of the body"),
    ("DICOM", "Standard file format for medical images, with metadata"),
    ("Hounsfield Unit (HU)", "CT density scale: air −1000, water 0, bone +1000"),
    ("Windowing", "Clipping HU to a range and mapping it to 0–255 grey levels"),
    ("Slice thickness / spacing", "Distance between consecutive slices (mm)"),
    ("Nodule", "Small round lesion in the lung, 3–30 mm; possibly early cancer"),
    ("Juxta-pleural / ground-glass", "Nodule touching the chest wall / faint hazy nodule"),
    ("LIDC-IDRI", "Public lung CT dataset with up to 4 radiologists' outlines per nodule"),
    ("Consensus mask (50%)", "Pixel is nodule if at least half of the annotating radiologists marked it"),
    ("Ground truth", "The reference answer (the consensus mask)"),
    ("Inter-observer variability", "Disagreement between radiologists; limits the achievable Dice")]))
A(gl("Data science", [
    ("Train / validation / test", "Learn on train, choose settings and checkpoints on validation, report once on test"),
    ("Patient-wise split", "All slices of a patient in one split; prevents leakage"),
    ("Data leakage", "Test information sneaking into training, which inflates scores"),
    ("Class imbalance", "Nodule pixels are under 0.1%; handled by Dice loss and nodule-centred crops"),
    ("Augmentation", "Random flips, rotations, noise etc. to create training variety"),
    ("Normalisation", "Rescaling inputs to mean about 0 and spread about 1"),
    ("2.5D", "Neighbouring slices stacked as channels of a 2D input")]))
A(gl("Deep learning", [
    ("Parameter / weight", "A learned number inside the network"),
    ("Convolution", "Sliding filter that detects local patterns"),
    ("Stride / padding", "Filter step size / zeros added at borders"),
    ("Activation (GELU, SiLU)", "Non-linear function applied after a layer"),
    ("BatchNorm / LayerNorm", "Rescale activations for stable training"),
    ("Residual connection", "Output = input + learned change"),
    ("Encoder / decoder", "Shrinks the image to understand it / enlarges it back to a mask"),
    ("Skip connection", "Passes high-resolution detail from encoder to decoder"),
    ("Feature map / token", "Intermediate image of features / one pixel's feature vector"),
    ("Logit", "Raw output before the sigmoid"),
    ("Sigmoid", "Squashes a logit to a probability in (0,1)"),
    ("Epoch / batch / step", "One pass over the data / images per update / one update"),
    ("Loss", "Number measuring error; minimised during training"),
    ("Backpropagation", "Chain-rule computation of gradients"),
    ("Optimiser (AdamW)", "Rule for updating weights from gradients"),
    ("Learning rate / warm-up / cosine", "Update size / gentle start / smooth decay"),
    ("Weight decay", "Penalty that keeps weights small to reduce overfitting"),
    ("Overfitting", "Memorising training data instead of generalising"),
    ("DropPath", "Randomly skipping residual branches during training"),
    ("Mixed precision (bf16)", "Using 16-bit numbers for speed where safe"),
    ("Gradient clipping", "Capping gradient size to prevent instability"),
    ("Checkpoint", "Saved model weights (best.pt / last.pt)"),
    ("Pretraining (ImageNet)", "Starting from weights learned on natural images"),
    ("Inference", "Using the trained model to predict")]))
A(gl("Transformers and Mamba", [
    ("Self-attention", "Each token computes a weighted mix of all others using Q·K similarity"),
    ("Query / Key / Value", "What I look for / what I offer / what I pass on"),
    ("Multi-head", "Several attention computations in parallel on channel groups"),
    ("Window attention (W-MSA)", "Attention only inside 8×8 windows; linear cost"),
    ("Shifted windows (SW-MSA)", "Shift by 4 px so windows exchange information"),
    ("Relative position bias", "Learned bias depending on the offset between tokens"),
    ("Patch merging", "2×2 tokens → 1 token, halving size and doubling channels"),
    ("State-space model (SSM)", "h′ = Ah + Bu, y = Ch + Du: a system with memory"),
    ("Hidden state", "Running memory vector carried along the scan (size 8)"),
    ("Discretisation (ZOH)", "Ā = exp(ΔA): converts the continuous system to steps"),
    ("Δ (step size)", "Input-dependent; controls forgetting versus remembering"),
    ("Selective scan", "SSM whose Δ, B, C depend on the input; the core of Mamba"),
    ("SS2D / VSS block", "4-direction 2D selective scan / the block that wraps it"),
    ("Linear complexity", "Cost grows proportionally with the number of tokens N (versus N² for attention)")]))
A(gl("Inference pipeline", [
    ("Post-processing", "Cleaning the predicted mask after the model (threshold, blob removal, masks)"),
    ("Connected component / blob", "A group of touching predicted pixels; one candidate nodule"),
    ("Test-time augmentation (TTA)", "Average predictions over flipped copies of the input"),
    ("Two-stage / cascade", "Detect on the full slice, then refine each detection on a zoomed patch"),
    ("Ablation", "Adding or removing one component at a time to measure its effect"),
    ("Tuning on validation", "Choosing settings on validation data so the test result stays unbiased")]))
A(gl("Evaluation", [
    ("TP / FP / FN / TN", "Correct nodule / false alarm / miss / correct background pixel"),
    ("Dice (DSC, F1)", "2TP/(2TP+FP+FN); overlap score 0–1"),
    ("IoU (Jaccard)", "TP/(TP+FP+FN)"),
    ("Precision", "Of predicted nodule pixels, share correct"),
    ("Recall / sensitivity", "Of true nodule pixels, share found"),
    ("False-alarm slice rate", "Share of empty slices where the model predicts a nodule"),
    ("Params / GMACs", "Model size / computation per image"),
    ("Full-slice versus crop protocol", "Find + outline on the whole slice versus outline only on a patch centred on the known nodule")]))

# ------------------------------------------------------------------ 19
A("""<h2>19. Likely viva questions</h2>
<dl class="qa">
<dt>Why crop the image instead of using the full slice?</dt><dd>In practice segmentation follows detection: a radiologist or detector marks the nodule and the network outlines it in a small ROI. Most LIDC-IDRI papers report this protocol, so our numbers are directly comparable. The full-slice version is our future scope and is already prototyped.</dd>
<dt>Isn't cropping cheating, since you know where the nodule is?</dt><dd>It is a different task (outlining, not finding), and we say so clearly. All four models get exactly the same patches, and the split is by patient, so the comparison is fair.</dd>
<dt>Why did crop training take under an hour?</dt><dd>Patches are 16× smaller than full slices, only nodule slices are used, and batches are larger.</dd>
<dt>Swin-UNet scores higher. Why is your model useful?</dt><dd>It keeps 96% of Swin-UNet's Dice with 13× fewer parameters, about 40× less computation and no pre-training. For a tool run once per detected nodule on ordinary hardware, that trade-off matters.</dd>
<dt>Why combine Swin and Mamba, not use only one?</dt><dd>Window attention is excellent for local texture at high resolution. Mamba gives global context at linear cost at low resolution. Using each where it is cheapest keeps the model small.</dd>
<dt>What exactly is Mamba?</dt><dd>A selective state-space model: a recurrence h<sub>t</sub> = Āh<sub>t−1</sub> + B̄u<sub>t</sub> whose parameters depend on the input, scanned over the image in four directions.</dd>
<dt>What does 2.5D add?</dt><dd>3D context for 288 extra parameters: +0.030 Dice on crops and +0.039 on full slices.</dd>
<dt>What happens on full slices?</dt><dd>Our 2.5D model is the best of the four (Dice 0.708 raw, 0.723 with flip-averaging; false alarms 12.1% → 6.9%), ahead of U-Net (0.696) and Swin-UNet (0.670).</dd>
<dt>Why is your Dice lower than papers reporting 0.89?</dt><dd>Some use slice-wise splits (neighbouring slices of one patient in both train and test), heavier models, or different consensus levels. Ours is patient-wise, and the radiologists' own agreement is about 0.85–0.88.</dd>
<dt>How do you know there is no leakage?</dt><dd>The split is by patient, checkpoints were selected on validation only, and the test set was used once.</dd>
</dl>""")

css = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: 'Segoe UI', Calibri, Arial, sans-serif; font-size: 10.3pt; line-height: 1.45; color: #1d2330; }
h1 { font-size: 26pt; line-height: 1.15; margin: 8px 0 10px; color: #13294b; }
h2 { font-size: 15pt; color: #13294b; border-bottom: 2px solid #c9d6ea; padding-bottom: 3px; margin: 22px 0 8px; break-after: avoid; }
h3 { font-size: 11.5pt; color: #2a4a7f; margin: 14px 0 4px; break-after: avoid; }
p { margin: 4px 0 7px; } ul, ol { margin: 4px 0 8px; padding-left: 22px; } li { margin: 2px 0; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #eef2f8; padding: 0 3px; border-radius: 3px; }
.cover { break-after: page; padding-top: 30px; }
.kicker { color: #8a2432; font-weight: 600; letter-spacing: .5px; text-transform: uppercase; font-size: 9.5pt; }
.sub { font-size: 12pt; color: #44506a; max-width: 92%; }
.kpi { width: 100%; margin: 22px 0; border-collapse: separate; border-spacing: 8px; }
.kpi td { background: #f0f4fa; border-radius: 8px; padding: 12px; text-align: center; width: 25%; }
.kpi b { display: block; font-size: 20pt; color: #13294b; } .kpi span { font-size: 9pt; color: #5a6680; }
.toc-h { font-weight: 700; margin-top: 18px; color: #13294b; } .toc { columns: 2; font-size: 10.5pt; }
.eq { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; background: #f6f8fc; border-left: 3px solid #2a4a7f;
      padding: 6px 12px; margin: 6px 0 8px; font-family: 'Cambria Math', Cambria, 'Times New Roman', serif; font-size: 11pt; break-inside: avoid; }
.eq .en { font-family: 'Segoe UI', Arial, sans-serif; font-size: 8.5pt; color: #5a6680; text-align: right; }
.note, .warn { padding: 7px 11px; border-radius: 6px; margin: 8px 0; break-inside: avoid; }
.note { background: #eaf3ec; border-left: 3px solid #2e7d4a; } .warn { background: #fdf3e4; border-left: 3px solid #c77d12; }
table.t { border-collapse: collapse; margin: 6px 0 10px; font-size: 9.4pt; break-inside: avoid; width: 100%; }
table.t td { border-bottom: 1px solid #dde3ee; padding: 3px 7px; vertical-align: top; }
table.t tr.h td { background: #13294b; color: #fff; font-weight: 600; } table.t tr.hl td { background: #fbeef0; }
table.gl td:first-child { width: 30%; } table.gl { break-inside: auto; }
figure { margin: 8px 0 12px; text-align: center; break-inside: avoid; } figure img { max-width: 100%; }
figcaption { font-size: 8.8pt; color: #5a6680; margin-top: 3px; font-style: italic; }
dl.qa dt { font-weight: 700; color: #13294b; margin-top: 7px; } dl.qa dd { margin: 2px 0 0 14px; }
"""
Path(sys.argv[1]).write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>ROI Version Study Guide</title><style>{css}</style></head><body>{''.join(H)}</body></html>", encoding="utf-8")
print("ok")
