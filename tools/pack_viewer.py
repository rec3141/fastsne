#!/usr/bin/env python3
"""Pack embeddings + colour layers for viewer.html.

    pack_viewer.py out.fsv --emb fastsne=emb.tsv [--emb other=table.tsv:tsne_x,tsne_y ...]
                           [--layer genus=cats.tsv:genus] [--layer gc=table.tsv:gc~num] [--layer depth=table.tsv:depth~log]
                           [--title T] [--page-size N] [--kind FILE:COL]
    pack_viewer.py emb.tsv out.fsv --color cats.tsv [--col NAME]       (old form: one embedding, one layer)

--emb NAME=FILE[:XCOL,YCOL]   tsv with an id column (first column) and coordinates (default columns x, y).
                              The first embedding fixes the point set; others are joined by id.
--layer NAME=FILE:COL[~num|~log]  a colouring. Categorical by default (empty, "NA" and missing are unlabelled;
                              names starting with "(" get no colour of their own). ~num / ~log bins a numeric
                              column into 64 linear / logarithmic steps on a colour ramp.

--page-size N  write OUT.json plus OUT.p000.fsv, OUT.p001.fsv, ... of N points each (OUT given as X.json or X).
              viewer.html?data=X.json loads them one after another and draws each as it arrives, so very large
              sets need not fit one browser allocation. Every page carries the full header (with "total").
--kind FILE:COL  rows whose COL is "reference" (points embedded directly) and the rest (points projected onto them,
              e.g. fastsne --project) go to separate pages, reference pages first; the viewer can hide projected
              pages. Implies paging (default page size 25M).

Format FSV2: b"FSV2", uint32 header length, JSON header (padded to 4 bytes), then per embedding uint16 x[n],
uint16 y[n] (quantised over its bounding box), then per layer uint16 cat[n] (0 = unlabelled). Points are
shuffled once, so the viewer can draw grey points first and coloured points in random order on top.
"""
import argparse
import colorsys
import json
import struct
import sys

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("pos", nargs="*")
ap.add_argument("--emb", action="append", default=[])
ap.add_argument("--layer", action="append", default=[])
ap.add_argument("--color")
ap.add_argument("--col")
ap.add_argument("--title", default="")
ap.add_argument("--page-size", type=int, default=0)
ap.add_argument("--kind")
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
    return to_hex(colorsys.hsv_to_rgb(h, s, v))


def to_hex(rgb):
    return "#" + "".join(f"{int(round(min(max(c, 0.0), 1.0) * 255)):02x}" for c in rgb)


def turbo(x):
    """Google's polynomial approximation of the turbo colour map (no matplotlib needed)"""
    r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
    g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
    b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
    return r, g, b


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
        nref = None
        if a.kind:
            # reference rows first, each group keeps its shuffled order
            kpath, _, kcol = a.kind.rpartition(":")
            isref = (table(kpath).reindex(ids)[kcol].astype("string") == "reference").to_numpy(dtype=bool, na_value=False)
            ids = ids[np.r_[np.flatnonzero(isref), np.flatnonzero(~isref)]]
            nref = int(isref.sum())
            a.page_size = a.page_size or 25_000_000
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
        counts = np.bincount(code, minlength=nb + 1)
        cats = [{"name": "missing", "count": int(counts[0]), "color": "#474747"}]
        for i in range(nb):
            nm = f"{fmt(edges[i])} – {fmt(edges[i + 1])}"
            if i == 0: nm = "≤ " + fmt(edges[1])
            if i == nb - 1: nm = "≥ " + fmt(edges[i])
            cats.append({"name": nm, "count": int(counts[i + 1]), "color": to_hex(turbo(0.05 + 0.9 * i / (nb - 1)))})
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

def write_fsv(path, lo, hi):
    """points lo..hi-1 as one FSV2 file; category counts in the header always cover all n points"""
    hdr = json.dumps({"n": hi - lo, "total": n, "title": a.title,
                      "embeddings": [{"name": e["name"], "bounds": e["bounds"]} for e in embs],
                      "layers": [{"name": l["name"], "ordinal": l["ordinal"], "categories": l["categories"]} for l in layers]}).encode()
    hdr += b" " * (-len(hdr) % 4)
    with open(path, "wb") as f:
        f.write(b"FSV2")
        f.write(struct.pack("<I", len(hdr)))
        f.write(hdr)
        for e in embs:
            e["q"][lo:hi, 0].copy().tofile(f)
            e["q"][lo:hi, 1].copy().tofile(f)
        for l in layers:
            l["code"][lo:hi].astype(np.uint16).tofile(f)


if a.page_size > 0:
    import os
    stem = out[:-5] if out.endswith(".json") else out

    def split(lo, hi):
        k = max(1, -(-(hi - lo) // a.page_size))
        e = np.linspace(lo, hi, k + 1).round().astype(int)
        return [(int(p), int(q)) for p, q in zip(e[:-1], e[1:]) if q > p]
    ranges = [(r, "") for r in split(0, n)] if nref is None else \
        [(r, "reference") for r in split(0, nref)] + [(r, "projected") for r in split(nref, n)]
    pages = [f"{stem}.p{k:03d}.fsv" for k in range(len(ranges))]
    entries = []
    for pg, ((lo, hi), kind) in zip(pages, ranges):
        write_fsv(pg, lo, hi)
        entries.append({"file": os.path.basename(pg), "kind": kind} if kind else os.path.basename(pg))
    with open(stem + ".json", "w") as f:
        json.dump({"format": "FSV2 pages", "total": n, "pages": entries}, f)
    out = f"{stem}.json + {len(pages)} pages"
else:
    write_fsv(out, 0, n)
print(f"wrote {out}: {n} points, {len(embs)} embeddings, {len(layers)} layers "
      f"({', '.join(l['name'] + ':' + str(len(l['categories']) - 1) for l in layers)})")
