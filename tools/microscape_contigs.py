#!/usr/bin/env python3
"""Download and decode the contig table of a microscape.app project into one TSV.

    microscape_contigs.py https://microscape.app/marine-coassembly-flye/ contigs.tsv.gz [--cache DIR]

The dashboard serves data/contig_explorer_manifest.json(.gz) plus chunked contig_explorer.part-NNN.json(.gz)
files: coded columns index into the manifest's vocab, and ids are delta-encoded integers with a prefix.
Output columns: id, then every column in the manifest (length, gc, depth, taxonomy per tool and rank, bins,
tsne/umap/pca coordinates, ...).
"""
import argparse
import gzip
import json
import os
import urllib.request

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("out")
ap.add_argument("--cache", default=".")
a = ap.parse_args()
base = a.url.rstrip("/") + "/data/"
os.makedirs(a.cache, exist_ok=True)


def get(name):
    path = os.path.join(a.cache, name + ".gz")
    if not os.path.exists(path):
        urllib.request.urlretrieve(base + name + ".gz", path)
    return json.load(gzip.open(path))


m = get("contig_explorer_manifest.json")
voc, enc = m["vocab"], m["id_encoding"]
parts = []
for ch in m["chunks"]:
    c = get(ch["file"])
    ids = np.cumsum(c["ids"])
    df = pd.DataFrame({"id": [f"{enc['prefix']}{i:0{enc['width']}d}" for i in ids]})
    for k, v in c["columns"].items():
        df[k] = np.array(voc[k], dtype=object)[np.array(v)] if k in voc else v
    parts.append(df)
    print(f"{ch['file']}: {len(df)} contigs", flush=True)
d = pd.concat(parts, ignore_index=True)
d.to_csv(a.out, sep="\t", index=False)
print(f"wrote {a.out}: {len(d)} contigs x {d.shape[1] - 1} columns")
