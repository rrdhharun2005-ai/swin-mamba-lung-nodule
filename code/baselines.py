"""Baseline segmentation models for the comparative study (same input/output contract as SwinMambaUNet).

- UNet      : Ronneberger et al. 2015, 4 down/up levels, base 64 channels (~31 M params).
- SwinUNet  : Cao et al. 2022 (pure Swin Transformer U-Net: patch embed /4, Swin encoder, patch-expanding
              decoder, x4 final expand). embed 96, depths [2,2,2,2], heads [3,6,12,24] (~27 M params).
              Window size 8 instead of the paper's 7 so 256/512 inputs divide evenly; trained from scratch
              (no ImageNet pre-training) to match the other models.
All models: forward(x: B,in_ch,H,W) -> logits B,1,H,W.  H, W divisible by 32.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import PatchMerging, SwinBlock, count_params


# ----------------------------------------------------------------------------- U-Net
class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, in_ch=1, num_classes=1, base=64):
        super().__init__()
        c = [base * 2 ** i for i in range(5)]  # 64,128,256,512,1024
        self.inc = DoubleConv(in_ch, c[0])
        self.downs = nn.ModuleList(DoubleConv(c[i], c[i + 1]) for i in range(4))
        self.ups = nn.ModuleList(nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in reversed(range(4)))
        self.up_convs = nn.ModuleList(DoubleConv(c[i] * 2, c[i]) for i in reversed(range(4)))
        self.head = nn.Conv2d(c[0], num_classes, 1)

    def forward(self, x):
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(F.max_pool2d(skips[-1], 2)))
        y = skips.pop()
        for up, conv in zip(self.ups, self.up_convs):
            y = conv(torch.cat([up(y), skips.pop()], 1))
        return self.head(y)


# ----------------------------------------------------------------------------- Swin-UNet
class PatchExpand(nn.Module):
    """B,H,W,C -> B,2H,2W,C/scale_div  (linear then pixel-shuffle style rearrange)"""

    def __init__(self, dim, scale=2, out_dim=None):
        super().__init__()
        self.scale = scale
        self.out_dim = out_dim or dim // 2
        self.expand = nn.Linear(dim, self.out_dim * scale * scale, bias=False)
        self.norm = nn.LayerNorm(self.out_dim)

    def forward(self, x):
        B, H, W, _ = x.shape
        s, c = self.scale, self.out_dim
        x = self.expand(x).view(B, H, W, s, s, c).permute(0, 1, 3, 2, 4, 5).reshape(B, H * s, W * s, c)
        return self.norm(x)


class SwinUNet(nn.Module):
    def __init__(self, in_ch=1, num_classes=1, embed=96, depths=(2, 2, 2, 2), heads=(3, 6, 12, 24),
                 window=8, drop_path=0.1):
        super().__init__()
        dims = [embed * 2 ** i for i in range(4)]  # 96,192,384,768
        self.patch_embed = nn.Conv2d(in_ch, embed, 4, stride=4)
        self.embed_norm = nn.LayerNorm(embed)
        dpr = torch.linspace(0, drop_path, sum(depths)).tolist()

        def stage(d, n, h, dp):
            return nn.Sequential(*[SwinBlock(d, h, window, shift=(j % 2 == 1), drop_path=dp[j]) for j in range(n)])

        self.enc, self.merges, i = nn.ModuleList(), nn.ModuleList(), 0
        for s in range(4):
            self.enc.append(stage(dims[s], depths[s], heads[s], dpr[i:i + depths[s]]))
            i += depths[s]
            if s < 3:
                self.merges.append(PatchMerging(dims[s]))
        self.bottleneck_norm = nn.LayerNorm(dims[3])

        self.expands, self.concat_proj, self.dec = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for s in (2, 1, 0):
            self.expands.append(PatchExpand(dims[s + 1]))
            self.concat_proj.append(nn.Linear(2 * dims[s], dims[s]))
            self.dec.append(stage(dims[s], depths[s], heads[s], [0.0] * depths[s]))
        self.dec_norm = nn.LayerNorm(embed)
        self.final_expand = PatchExpand(embed, scale=4, out_dim=embed)
        self.head = nn.Conv2d(embed, num_classes, 1)
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
        y = self.embed_norm(self.patch_embed(x).permute(0, 2, 3, 1))  # B,H/4,W/4,C
        skips = []
        for s in range(4):
            y = self.enc[s](y)
            if s < 3:
                skips.append(y)
                y = self.merges[s](y)
        y = self.bottleneck_norm(y)
        for expand, proj, dec in zip(self.expands, self.concat_proj, self.dec):
            y = dec(proj(torch.cat([expand(y), skips.pop()], -1)))
        y = self.final_expand(self.dec_norm(y))  # B,H,W,C
        return self.head(y.permute(0, 3, 1, 2).contiguous())


PRETRAINED_SWIN_T = __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "pretrained",
                                                "swin_tiny_patch4_window7_224.pth")


class SwinUNetPretrained(SwinUNet):
    """Swin-UNet as in Cao et al. 2022: encoder = Swin-T (depths 2,2,6,2) initialised from ImageNet-1k weights,
    decoder stages initialised from the matching encoder stages (as in the official code).
    Adaptations: RGB patch-embed kernel summed to in_ch=1 (or kept for 3), relative-position bias tables
    bicubically resized from window 7 to window 8."""

    def __init__(self, in_ch=1, num_classes=1, weights=PRETRAINED_SWIN_T):
        super().__init__(in_ch, num_classes, embed=96, depths=(2, 2, 6, 2), heads=(3, 6, 12, 24), window=8,
                         drop_path=0.2)
        import os
        if weights and os.path.exists(weights):
            self.load_swin_t(weights, in_ch)

    def load_swin_t(self, path, in_ch):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)

        def bias(t):  # (13*13, h) -> (15*15, h)
            h = t.shape[1]
            t = t.T.reshape(1, h, 13, 13)
            return F.interpolate(t, size=(15, 15), mode="bicubic", align_corners=False).reshape(h, 225).T

        def load_block(blk, pre):
            blk.norm1.weight.data.copy_(sd[pre + "norm1.weight"]); blk.norm1.bias.data.copy_(sd[pre + "norm1.bias"])
            blk.norm2.weight.data.copy_(sd[pre + "norm2.weight"]); blk.norm2.bias.data.copy_(sd[pre + "norm2.bias"])
            for a, b in (("qkv", "qkv"), ("proj", "proj")):
                getattr(blk.attn, a).weight.data.copy_(sd[pre + f"attn.{b}.weight"])
                getattr(blk.attn, a).bias.data.copy_(sd[pre + f"attn.{b}.bias"])
            blk.attn.bias_table.data.copy_(bias(sd[pre + "attn.relative_position_bias_table"]))
            for f in ("fc1", "fc2"):
                getattr(blk.mlp, f).weight.data.copy_(sd[pre + f"mlp.{f}.weight"])
                getattr(blk.mlp, f).bias.data.copy_(sd[pre + f"mlp.{f}.bias"])

        w = sd["patch_embed.proj.weight"]  # 96,3,4,4
        self.patch_embed.weight.data.copy_(w.sum(1, keepdim=True) if in_ch == 1 else w[:, :in_ch])
        self.patch_embed.bias.data.copy_(sd["patch_embed.proj.bias"])
        self.embed_norm.weight.data.copy_(sd["patch_embed.norm.weight"])
        self.embed_norm.bias.data.copy_(sd["patch_embed.norm.bias"])
        for s in range(4):
            for j, blk in enumerate(self.enc[s]):
                load_block(blk, f"layers.{s}.blocks.{j}.")
            if s < 3:
                self.merges[s].norm.weight.data.copy_(sd[f"layers.{s}.downsample.norm.weight"])
                self.merges[s].norm.bias.data.copy_(sd[f"layers.{s}.downsample.norm.bias"])
                self.merges[s].red.weight.data.copy_(sd[f"layers.{s}.downsample.reduction.weight"])
        self.bottleneck_norm.weight.data.copy_(sd["norm.weight"]); self.bottleneck_norm.bias.data.copy_(sd["norm.bias"])
        for dec, s in zip(self.dec, (2, 1, 0)):  # decoder stage s <- encoder stage s (first blocks)
            for j, blk in enumerate(dec):
                load_block(blk, f"layers.{s}.blocks.{j}.")
        self.loaded_pretrained = True


MODELS = {"unet": UNet, "swin_unet": SwinUNet, "swin_unet_pt": SwinUNetPretrained}


if __name__ == "__main__":
    for name, cls in MODELS.items():
        m = cls()
        print(f"{name}: {count_params(m) / 1e6:.2f} M", m(torch.randn(1, 1, 256, 256)).shape)
