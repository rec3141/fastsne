#!/usr/bin/env python3
"""Kraken per-read names -> genus table for render.py.

    kraken_genus.py file1.tsv [file2.tsv ...] > genus.tsv   (or @list.txt; input: read_id<TAB>taxon name)

Handles names with spaces or underscores, strips "(taxid N)", keeps "Candidatus X" as a genus, drops
"unclassified" prefixes. Names that look like ranks above genus (Bacteria, Pseudomonadota, Flavobacteriaceae,
Gammaproteobacteria, ...) become "(above genus)", which render.py --cat never colours.
"""
import re
import sys

files = []
for a in sys.argv[1:]:
    files += [l.strip() for l in open(a[1:])] if a.startswith("@") else [a]
HIGHER = {"Bacteria", "Archaea", "Eukaryota", "Viruses", "cellular", "root", "Bacilli", "Clostridia"}
HIGHER_SUFFIX = ("aceae", "ales", "ota", "ati", "idae", "ineae", "proteobacteria", "mycetes", "bacteriia", "phyceae")
cache = {}
out = sys.stdout
out.write("id\tgenus\n")
for fn in files:
    for line in open(fn):
        p = line.rstrip("\n").split("\t")
        if len(p) < 2:
            continue
        name = p[1]
        g = cache.get(name)
        if g is None:
            t = re.sub(r"\s*\(taxid[ _]\d+\)\s*$", "", name.replace("_", " ")).split()
            while t and t[0] == "unclassified":
                t = t[1:]
            if not t:
                g = "unclassified"
            elif t[0] == "Candidatus" and len(t) > 1:
                g = "Ca. " + t[1]
            elif t[0] in ("cellular", "root") or (len(t) == 1 and (t[0] in HIGHER or t[0].endswith(HIGHER_SUFFIX))):
                g = "(above genus)"
            else:
                g = t[0]
            cache[name] = g
        out.write(f"{p[0]}\t{g}\n")
