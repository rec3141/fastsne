#!/usr/bin/env python3
"""Pack embeddings + colour layers for viewer.html.

    pack_viewer.py out.fsv --emb fastsne=emb.tsv [--emb other=table.tsv:tsne_x,tsne_y ...]
                           [--layer genus=cats.tsv:genus] [--layer gc=table.tsv:gc~num] [--layer depth=table.tsv:depth~log]
                           [--title T]
    pack_viewer.py emb.tsv out.fsv --color cats.tsv [--col NAME]       (old form: one embedding, one layer)

--emb NAME=FILE[:XCOL,YCOL]   tsv with an id column (first column) and coordinates (default columns x, y).
                              The first embedding fixes the point set; others are joined by id.
--layer NAME=FILE:COL[~num|~log]  a colouring. Categorical by default (empty, "NA" and missing are unlabelled;
                              names starting with "(" get no colour of their own). ~num / ~log bins a numeric
                              column into 64 linear / logarithmic steps on a colour ramp.

Format FSV2: b"FSV2", uint32 header length, JSON header (padded to 4 bytes), then per embedding uint16 x[n],
uint16 y[n] (quantised over its bounding box), then per layer uint16 cat[n] (0 = unlabelled). Points are
shuffled once, so the viewer can draw grey points first and coloured points in random order on top.
"""
import argparse
import json
import struct
import sys

import matplotlib
import matplotlib.colors
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("pos", nargs="*")
ap.add_argument("--emb", action="append", default=[])
ap.add_argument("--layer", action="append", default=[])
ap.add_argument("--color")
ap.add_argument("--col")
ap.add_argument("--title", default="")
a = ap.parse_args()
if len(a.pos) == 2:          # old form: emb.tsv out.fsv [--color f --col c]
    a.emb.insert(0, f"t-SNE={a.pos[0]}")
    if a.color:
        a.layer.insert(0, f"{a.col or 'category'}={a.color}:{a.col or ''}")
    out = a.pos[1]
elif len(a.pos) == 1:
    out = a.pos[0]
else:
    sys.exit("give OUT.fsv (or EMB.tsv OUT.fsv)")
if not a.emb:
    sys.exit("need at least one --emb")

_tables = {}


def table(path):
    if path not in _tables:
        t = pd.read_csv(path, sep="\t", dtype={0: str}, low_memory=False)
        t = t.rename(columns={t.columns[0]: "id"}).drop_duplicates("id").set_index("id")
        _tables[path] = t
    return _tables[path]


def vivid(k):
    h, s, v = (0.08 + k * 0.618034) % 1, 1.0 - 0.12 * (k % 2), 1.0 - 0.14 * ((k // 2) % 3)
    return matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb((h, s, v)))


# embeddings
ids = None
embs = []
for spec in a.emb:
    name, rest = spec.split("=", 1)
    path, _, cols = rest.partition(":")
    xc, yc = cols.split(",") if cols else ("x", "y")
    t = table(path)
    if ids is None:
        ids = t.index
        rng = np.random.default_rng(1)
        ids = ids[rng.permutation(len(ids))]
    xy = t.reindex(ids)[[xc, yc]].to_numpy(np.float64)
    ok = np.isfinite(xy).all(1)
    lo, hi = xy[ok].min(0), xy[ok].max(0)
    q = np.round((xy - lo) / np.maximum(hi - lo, 1e-12) * 65534).clip(0, 65534)
    q[~ok] = 65535                                  # missing: drawn off-screen by the viewer
    embs.append({"name": name, "bounds": [lo[0], hi[0], lo[1], hi[1]], "q": q.astype(np.uint16)})
n = len(ids)

# layers
layers = []
for spec in a.layer:
    name, rest = spec.split("=", 1)
    path, _, col = rest.rpartition(":")
    mode = "cat"
    if "~" in col:
        col, mode = col.split("~")
    t = table(path)
    col = col or t.columns[0]
    v = t.reindex(ids)[col]
    if mode in ("num", "log"):
        x = pd.to_numeric(v, errors="coerce").to_numpy(np.float64)
        ok = np.isfinite(x) & ((x > 0) if mode == "log" else True)
        xs = np.log10(x[ok]) if mode == "log" else x[ok]
        lo, hi = np.percentile(xs, [0.5, 99.5])
        nb = 64
        b = np.clip(((xs - lo) / max(hi - lo, 1e-12) * nb).astype(int), 0, nb - 1)
        code = np.zeros(n, np.int64)
        code[ok] = b + 1
        edges = lo + (hi - lo) * np.arange(nb + 1) / nb
        fmt = (lambda e: f"{10 ** e:,.3g}") if mode == "log" else (lambda e: f"{e:,.3g}")
        cmap = matplotlib.colormaps["turbo"]
        counts = np.bincount(code, minlength=nb + 1)
        cats = [{"name": "missing", "count": int(counts[0]), "color": "#474747"}]
        for i in range(nb):
            nm = f"{fmt(edges[i])} – {fmt(edges[i + 1])}"
            if i == 0: nm = "≤ " + fmt(edges[1])
            if i == nb - 1: nm = "≥ " + fmt(edges[i])
            cats.append({"name": nm, "count": int(counts[i + 1]), "color": matplotlib.colors.to_hex(cmap(0.05 + 0.9 * i / (nb - 1)))})
        layers.append({"name": name, "ordinal": True, "categories": cats, "code": code})
    else:
        v = v.astype("string").str.strip()
        v = v.mask(v.isin(["", "NA", "nan", "None", "unclassified"]))
        vc = v.value_counts()
        code = v.map({t: i + 1 for i, t in enumerate(vc.index)}).fillna(0).to_numpy().astype(np.int64)
        cats = [{"name": "unlabelled", "count": int((code == 0).sum()), "color": "#474747"}]
        k = 0
        for t_, c_ in zip(vc.index, vc.values):
            if str(t_).startswith("("):
                cats.append({"name": str(t_), "count": int(c_), "color": "#808080"})
            else:
                cats.append({"name": str(t_), "count": int(c_), "color": vivid(k)}); k += 1
        layers.append({"name": name, "ordinal": False, "categories": cats, "code": code})
    if len(layers[-1]["categories"]) > 65535:
        sys.exit(f"layer {name}: too many categories")
if not layers:
    layers.append({"name": "none", "ordinal": False, "categories": [{"name": "all", "count": n, "color": "#c8d2e6"}],
                   "code": np.zeros(n, np.int64)})

hdr = json.dumps({"n": n, "title": a.title,
                  "embeddings": [{"name": e["name"], "bounds": e["bounds"]} for e in embs],
                  "layers": [{"name": l["name"], "ordinal": l["ordinal"], "categories": l["categories"]} for l in layers]}).encode()
hdr += b" " * (-len(hdr) % 4)
with open(out, "wb") as f:
    f.write(b"FSV2")
    f.write(struct.pack("<I", len(hdr)))
    f.write(hdr)
    for e in embs:
        e["q"][:, 0].copy().tofile(f)
        e["q"][:, 1].copy().tofile(f)
    for l in layers:
        l["code"].astype(np.uint16).tofile(f)
print(f"wrote {out}: {n} points, {len(embs)} embeddings, {len(layers)} layers "
      f"({', '.join(l['name'] + ':' + str(len(l['categories']) - 1) for l in layers)})")
