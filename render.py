#!/usr/bin/env python3
"""Render a fastsne embedding to PNG (density-shaded, optionally coloured by a per-id value).

render.py emb.tsv out.png [--color file.tsv [--col NAME]] [--cat] [--px 1600]
--color: tsv with id in first column; numeric -> viridis-like map of the mean per pixel,
         --cat -> categorical (top --ntop categories by count; other categories light grey, unlabelled dark grey).
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("emb")
ap.add_argument("out")
ap.add_argument("--color")
ap.add_argument("--col")
ap.add_argument("--cat", action="store_true")
ap.add_argument("--ntop", type=int, default=12, help="categories coloured with --cat (rest: light grey)")
ap.add_argument("--px", type=int, default=1600)
ap.add_argument("--title", default="")
a = ap.parse_args()

e = pd.read_csv(a.emb, sep="\t", dtype={"id": str}, engine="c")
x, y = e.x.to_numpy(), e.y.to_numpy()
lo = np.percentile(np.c_[x, y], 0.05, axis=0)
hi = np.percentile(np.c_[x, y], 99.95, axis=0)
span = (hi - lo).max() * 1.04
c = (hi + lo) / 2
x0, y0 = c - span / 2
px = a.px
ix = np.clip(((x - x0) / span * px).astype(int), 0, px - 1)
iy = np.clip(((y - y0) / span * px).astype(int), 0, px - 1)
lin = (px - 1 - iy) * px + ix
cnt = np.bincount(lin, minlength=px * px).astype(float)
alpha = np.log1p(cnt) / np.log1p(np.percentile(cnt[cnt > 0], 99.5))
alpha = np.clip(alpha, 0, 1) ** 0.6
alpha[cnt > 0] = np.maximum(alpha[cnt > 0], 0.25)

bg = np.array([0.06, 0.07, 0.11])
if a.color:
    cdf = pd.read_csv(a.color, sep="\t", dtype={0: str}, comment=None)
    cdf = cdf.rename(columns={cdf.columns[0]: "id"}).drop_duplicates("id")
    col = a.col or cdf.columns[1]
    v = e[["id"]].merge(cdf[["id", col]], on="id", how="left")[col]
    if a.cat:
        vc = v.value_counts()
        top = [t for t in vc.index if not str(t).startswith("(")][:a.ntop]
        base = ["#f6c344", "#35d0e8", "#3ad29f", "#fb7a6e", "#a88bfa", "#f5923e", "#60a5fa",
                "#e879f9", "#a3e635", "#fda4af", "#5eead4", "#fde68a"]
        pal = [matplotlib.colors.to_rgb(p) for p in base]
        k = 0
        while len(pal) < len(top):  # golden-ratio hues, alternating lightness, for the rest
            h = (0.11 + k * 0.618034) % 1
            pal.append(matplotlib.colors.hsv_to_rgb((h, 0.55 + 0.25 * (k % 2), 0.98 - 0.2 * ((k // 2) % 2))))
            k += 1
        pal = np.array(pal[:len(top)])
        code = np.full(len(v), -1)
        for k, t in enumerate(top):
            code[(v == t).to_numpy()] = k
        rgb = np.zeros((px * px, 3))
        known = code >= 0
        rgbp = np.full((len(v), 3), 0.28)                 # unlabelled
        rgbp[(~known) & v.notna().to_numpy()] = 0.5        # labelled, not in the top list
        rgbp[known] = pal[code[known]]
        for ch in range(3):
            rgb[:, ch] = np.bincount(lin, weights=rgbp[:, ch], minlength=px * px)
        rgb /= np.maximum(cnt, 1)[:, None]
        labels = list(top)
    else:
        vv = v.to_numpy(dtype=float)
        ok = np.isfinite(vv)
        s = np.bincount(lin[ok], weights=vv[ok], minlength=px * px)
        n = np.bincount(lin[ok], minlength=px * px)
        m = s / np.maximum(n, 1)
        vlo, vhi = np.nanpercentile(vv[ok], [1, 99])
        rgb = plt.get_cmap("turbo")(np.clip((m - vlo) / (vhi - vlo), 0, 1))[:, :3]
        labels = None
else:
    rgb = np.tile(np.array([0.85, 0.9, 1.0]), (px * px, 1))
    labels = None
img = bg[None, :] * (1 - alpha[:, None]) + rgb * alpha[:, None]
img = img.reshape(px, px, 3)
fig = plt.figure(figsize=(px / 100, px / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(img, interpolation="nearest")
ax.axis("off")
if a.title:
    ax.text(0.01, 0.99, a.title, color="w", fontsize=14, va="top", transform=ax.transAxes)
if labels is not None:
    per_col = 25
    fs = 11 if len(labels) <= 12 else 9
    for k, t in enumerate(labels):
        ax.text(0.01 + (k // per_col) * 0.16, 0.955 - (k % per_col) * 0.019, str(t)[:28], color=pal[k],
                fontsize=fs, transform=ax.transAxes)
fig.savefig(a.out, facecolor=bg)
print("wrote", a.out, len(e), "points")
