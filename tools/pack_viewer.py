#!/usr/bin/env python3
"""Pack an embedding (+ optional categories) for viewer.html.

    pack_viewer.py emb.tsv out.fsv [--color cats.tsv [--col NAME]] [--title T]

emb.tsv: id, x, y (fastsne output). cats.tsv: id in the first column, category in --col (default: second).
Categories whose name starts with "(" (e.g. "(above genus)") are kept but not given their own colour.

Format: b"FSV1", uint32 header length, JSON header (padded to 4 bytes), then uint16 x[n], uint16 y[n],
uint16 cat[n]. x, y are quantised over the bounding box. Category 0 = unlabelled. Unlabelled and uncoloured
points are drawn first; coloured points follow in random order, so each pixel shows roughly its majority
category (drawing rare categories last would sprinkle them over everything).
"""
import argparse
import json
import struct

import matplotlib.colors
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("emb")
ap.add_argument("out")
ap.add_argument("--color")
ap.add_argument("--col")
ap.add_argument("--title", default="")
a = ap.parse_args()

e = pd.read_csv(a.emb, sep="\t", dtype={"id": str}, engine="c")
n = len(e)
if a.color:
    c = pd.read_csv(a.color, sep="\t", dtype=str)
    c = c.rename(columns={c.columns[0]: "id"}).drop_duplicates("id")
    col = a.col or c.columns[1]
    v = e[["id"]].merge(c[["id", col]], on="id", how="left")[col]
else:
    v = pd.Series([None] * n)
del e["id"]

vc = v.value_counts()
names = ["unlabelled"] + list(vc.index)
counts = [int(v.isna().sum())] + [int(x) for x in vc.values]
code = v.map({t: i + 1 for i, t in enumerate(vc.index)}).fillna(0).to_numpy().astype(np.int64)
if len(names) > 65535:
    raise SystemExit(f"{len(names)} categories; at most 65535 supported")

# same vivid palette as render.py, in rank order among coloured categories
colors = []
k = 0
for i, t in enumerate(names):
    if i == 0:
        colors.append("#474747")
    elif str(t).startswith("("):
        colors.append("#808080")
    else:
        h, s, val = (0.08 + k * 0.618034) % 1, 1.0 - 0.12 * (k % 2), 1.0 - 0.14 * ((k // 2) % 3)
        colors.append(matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb((h, s, val))))
        k += 1

# draw order: unlabelled, then uncoloured groups, then coloured points shuffled
grey = np.array([i == 0 or str(t).startswith("(") for i, t in enumerate(names)])
rng = np.random.default_rng(1)
key = np.where(grey[code], code.astype(np.float64) - 1e9, rng.random(n))
order = np.argsort(key, kind="stable")

x, y = e.x.to_numpy(np.float64)[order], e.y.to_numpy(np.float64)[order]
code = code[order]
xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
qx = np.round((x - xmin) / max(xmax - xmin, 1e-12) * 65535).astype(np.uint16)
qy = np.round((y - ymin) / max(ymax - ymin, 1e-12) * 65535).astype(np.uint16)

hdr = json.dumps({"n": n, "bounds": [xmin, xmax, ymin, ymax], "title": a.title,
                  "categories": [{"name": str(t), "count": cnt, "color": col}
                                 for t, cnt, col in zip(names, counts, colors)]}).encode()
hdr += b" " * (-len(hdr) % 4)
with open(a.out, "wb") as f:
    f.write(b"FSV1")
    f.write(struct.pack("<I", len(hdr)))
    f.write(hdr)
    qx.tofile(f)
    qy.tofile(f)
    code.astype(np.uint16).tofile(f)
print(f"wrote {a.out}: {n} points, {len(names)} categories")
