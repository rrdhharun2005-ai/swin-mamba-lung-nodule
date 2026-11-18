"""Lightweight Swin-Mamba U-Net for lung nodule segmentation.

Encoder (input H x W, single-channel CT slice):
    stem   : 2 strided convs                       -> H/2 (skip), H/4
    stage 1: Swin blocks (shifted-window attention) at H/4   - local texture
    stage 2: Swin blocks                            at H/8
    stage 3: VSS (Mamba / 2D selective scan) blocks at H/16  - global context, linear cost
    stage 4: VSS blocks                             at H/32
Decoder: conv U-Net decoder with skip connections back to full resolution.

The selective scan is implemented in pure PyTorch (chunked parallel scan), so no CUDA
extension (mamba-ssm) is needed and it runs on Windows / CPU / any GPU.
Input H and W must be divisible by 32.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(1 - self.p)
        return x * keep / (1 - self.p)


class Mlp(nn.Module):
    def __init__(self, dim, ratio=4.0):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1, self.act, self.fc2 = nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ----------------------------------------------------------------------------- Swin
def window_partition(x, ws):  # (B,H,W,C) -> (B*nW, ws*ws, C)
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, ws * ws, C)


def window_reverse(w, ws, B, H, W):  # inverse of window_partition
    C = w.shape[-1]
    w = w.view(B, H // ws, W // ws, ws, ws, C).permute(0, 1, 3, 2, 4, 5)
    return w.reshape(B, H, W, C)


def shifted_window_mask(H, W, ws, s, device):
    img = torch.zeros(1, H, W, 1, device=device)
    cnt = 0
    for h in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
        for w in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
            img[:, h, w, :] = cnt
            cnt += 1
    mw = window_partition(img, ws).squeeze(-1)  # nW, N
    m = mw[:, None, :] - mw[:, :, None]
    return m.masked_fill(m != 0, -100.0).masked_fill(m == 0, 0.0)  # nW, N, N


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, ws):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.bias_table = nn.Parameter(torch.zeros((2 * ws - 1) ** 2, heads))
        nn.init.trunc_normal_(self.bias_table, std=0.02)
        c = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij")).flatten(1)
        rel = (c[:, :, None] - c[:, None, :]).permute(1, 2, 0) + ws - 1
        self.register_buffer("rel_idx", rel[..., 0] * (2 * ws - 1) + rel[..., 1], persistent=False)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B_, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        bias = self.bias_table[self.rel_idx.view(-1)].view(N, N, -1).permute(2, 0, 1)  # h,N,N
        if mask is None:
            attn_mask = bias.unsqueeze(0)
        else:
            nW = mask.shape[0]
            attn_mask = (bias.unsqueeze(0) + mask.unsqueeze(1)).repeat(B_ // nW, 1, 1, 1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask.to(q.dtype))
        return self.proj(out.transpose(1, 2).reshape(B_, N, C))


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, ws=8, shift=False, drop_path=0.0, mlp_ratio=4.0):
        super().__init__()
        self.ws, self.shift = ws, shift
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, heads, ws)
        self.mlp = Mlp(dim, mlp_ratio)
        self.dp = DropPath(drop_path)
        self._mask_cache = {}

    def forward(self, x):  # B,H,W,C
        B, H, W, C = x.shape
        ws = min(self.ws, H, W)
        assert H % ws == 0 and W % ws == 0, "feature size must be divisible by window size"
        s = ws // 2 if (self.shift and min(H, W) > ws) else 0
        y = self.norm1(x)
        mask = None
        if s:
            y = torch.roll(y, (-s, -s), (1, 2))
            key = (H, W, ws, s, x.device)
            if key not in self._mask_cache:
                self._mask_cache[key] = shifted_window_mask(H, W, ws, s, x.device)
            mask = self._mask_cache[key]
        y = window_reverse(self.attn(window_partition(y, ws), mask), ws, B, H, W)
        if s:
            y = torch.roll(y, (s, s), (1, 2))
        x = x + self.dp(y)
        return x + self.dp(self.mlp(self.norm2(x)))


# ----------------------------------------------------------------------------- Mamba (SS2D)
def selective_scan(u, delta, A, Bm, Cm, Dp, chunk=8, min_log_decay=-4.0):
    """Chunked parallel selective scan:  h_t = exp(delta_t*A) h_{t-1} + delta_t*B_t*u_t,  y_t = C_t.h_t + D*u_t

    u, delta: (B,K,D,L)   A: (K,D,N)   Bm, Cm: (B,K,N,L)   Dp: (K,D)   ->  y: (B,K,D,L)
    Inside a chunk the recurrence is solved with cumulative sums in log space; chunks are
    then linked by a short sequential loop over L/chunk steps. Run in float32.
    Numerical safety: the per-step log decay is clamped to >= min_log_decay, so inside a chunk
    |cumsum| <= chunk*|min_log_decay| = 32 and exp(+-32) stays far from float32 overflow
    (a chunk of 16 with clamp -5 reached exp(80) and produced inf/NaN during training).
    """
    Bsz, K, D, L = u.shape
    N = A.shape[-1]
    pad = (-L) % chunk
    if pad:
        u, delta = F.pad(u, (0, pad)), F.pad(delta, (0, pad))
        Bm, Cm = F.pad(Bm, (0, pad)), F.pad(Cm, (0, pad))
    nc = (L + pad) // chunk

    la = (delta.unsqueeze(-1) * A[None, :, :, None, :]).clamp(min=min_log_decay)  # log decay B,K,D,L,N
    bx = (delta * u).unsqueeze(-1) * Bm.transpose(-1, -2).unsqueeze(2)  # input       B,K,D,L,N
    la = la.view(Bsz, K, D, nc, chunk, N)
    bx = bx.view(Bsz, K, D, nc, chunk, N)

    cum = la.cumsum(4)
    decay = torch.exp(cum)  # decay from chunk start to t
    h_intra = decay * torch.cumsum(bx * torch.exp(-cum), 4)  # state if chunk started at 0

    h = u.new_zeros(Bsz, K, D, N)
    starts = []
    for c in range(nc):
        starts.append(h)
        h = h_intra[:, :, :, c, -1] + decay[:, :, :, c, -1] * h
    h_all = h_intra + decay * torch.stack(starts, 3).unsqueeze(4)

    Cc = Cm.transpose(-1, -2).reshape(Bsz, K, 1, nc, chunk, N)
    y = (h_all * Cc).sum(-1).reshape(Bsz, K, D, nc * chunk)[..., :L]
    return y + u[..., :L] * Dp[None, :, :, None]


class SS2D(nn.Module):
    """2D selective scan over 4 directions (row, column, and their reverses)."""

    def __init__(self, dim, d_state=8, expand=2, d_conv=3):
        super().__init__()
        D, N, R, K = expand * dim, d_state, math.ceil(dim / 16), 4
        self.D, self.N, self.R = D, N, R
        self.in_proj = nn.Linear(dim, 2 * D, bias=False)
        self.conv = nn.Conv2d(D, D, d_conv, padding=d_conv // 2, groups=D)
        self.x_proj = nn.Parameter(torch.randn(K, R + 2 * N, D) * D ** -0.5)
        self.dt_proj = nn.Parameter(torch.randn(K, D, R) * R ** -0.5)
        dt = torch.exp(torch.rand(K, D) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        self.A_log = nn.Parameter(torch.log(torch.arange(1, N + 1).float()).repeat(K, D, 1))
        self.Ds = nn.Parameter(torch.ones(K, D))
        self.out_norm = nn.LayerNorm(D)
        self.out_proj = nn.Linear(D, dim, bias=False)

    def forward(self, x):  # B,H,W,C
        B, H, W, _ = x.shape
        x, z = self.in_proj(x).chunk(2, -1)
        x = F.silu(self.conv(x.permute(0, 3, 1, 2)))  # B,D,H,W
        x_hw = x.flatten(2)
        x_wh = x.transpose(2, 3).flatten(2)
        xs = torch.stack([x_hw, x_wh, x_hw.flip(-1), x_wh.flip(-1)], 1)  # B,4,D,L

        proj = torch.einsum("bkdl,kcd->bkcl", xs, self.x_proj)
        dt, Bm, Cm = proj.split([self.R, self.N, self.N], 2)
        dt = torch.einsum("bkrl,kdr->bkdl", dt, self.dt_proj)
        with torch.autocast(device_type=x.device.type, enabled=False):
            delta = F.softplus(dt.float() + self.dt_bias[None, :, :, None])
            A = -torch.exp(self.A_log.float())
            y = selective_scan(xs.float(), delta, A, Bm.float(), Cm.float(), self.Ds.float())

        y_wh = (y[:, 1] + y[:, 3].flip(-1)).view(B, -1, W, H).transpose(2, 3).flatten(2)
        y = y[:, 0] + y[:, 2].flip(-1) + y_wh  # B,D,L
        y = y.transpose(1, 2).reshape(B, H, W, -1).to(z.dtype)
        return self.out_proj(self.out_norm(y) * F.silu(z))


class VSSBlock(nn.Module):
    def __init__(self, dim, d_state=8, drop_path=0.0, mlp_ratio=2.0):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ss2d = SS2D(dim, d_state)
        self.mlp = Mlp(dim, mlp_ratio)
        self.dp = DropPath(drop_path)

    def forward(self, x):
        x = x + self.dp(self.ss2d(self.norm1(x)))
        return x + self.dp(self.mlp(self.norm2(x)))


# ----------------------------------------------------------------------------- U-Net
class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.red = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1)
        return self.red(self.norm(x))


def conv_bn_act(cin, cout, k=3, s=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.BatchNorm2d(cout), nn.GELU())


class UpBlock(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = nn.Sequential(conv_bn_act(cin + cskip, cout), conv_bn_act(cout, cout))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class SwinMambaUNet(nn.Module):
    def __init__(self, in_ch=1, num_classes=1, dims=(32, 64, 128, 256), depths=(2, 2, 2, 2),
                 heads=(2, 4, 8, 16), window=8, d_state=8, drop_path=0.1):
        super().__init__()
        self.stem1 = conv_bn_act(in_ch, dims[0] // 2, 3, 2)  # H/2
        self.stem2 = nn.Sequential(conv_bn_act(dims[0] // 2, dims[0], 3, 2),
                                   nn.Conv2d(dims[0], dims[0], 1))  # H/4
        dpr = torch.linspace(0, drop_path, sum(depths)).tolist()
        self.stages, self.norms, self.downs = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        i = 0
        for s, (d, n) in enumerate(zip(dims, depths)):
            if s < 2:
                blocks = [SwinBlock(d, heads[s], window, shift=(j % 2 == 1), drop_path=dpr[i + j]) for j in range(n)]
            else:
                blocks = [VSSBlock(d, d_state, drop_path=dpr[i + j]) for j in range(n)]
            i += n
            self.stages.append(nn.Sequential(*blocks))
            self.norms.append(nn.LayerNorm(d))
            self.downs.append(PatchMerging(d) if s < len(dims) - 1 else nn.Identity())

        self.up3 = UpBlock(dims[3], dims[2], dims[2])
        self.up2 = UpBlock(dims[2], dims[1], dims[1])
        self.up1 = UpBlock(dims[1], dims[0], dims[0])
        self.up0 = UpBlock(dims[0], dims[0] // 2, dims[0] // 2)
        self.head = nn.Sequential(conv_bn_act(dims[0] // 2, dims[0] // 2), nn.Conv2d(dims[0] // 2, num_classes, 1))
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        H, W = x.shape[-2:]
        s0 = self.stem1(x)  # B,C/2,H/2,W/2
        y = self.stem2(s0).permute(0, 2, 3, 1)  # B,H/4,W/4,C
        feats = []
        for stage, norm, down in zip(self.stages, self.norms, self.downs):
            y = stage(y)
            feats.append(norm(y).permute(0, 3, 1, 2).contiguous())
            y = down(y)
        f1, f2, f3, f4 = feats
        d = self.up3(f4, f3)
        d = self.up2(d, f2)
        d = self.up1(d, f1)
        d = self.up0(d, s0)
        d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)
        return self.head(d)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = SwinMambaUNet()
    print(f"params: {count_params(m) / 1e6:.2f} M")
    print(m(torch.randn(2, 1, 256, 256)).shape)
