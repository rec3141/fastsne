#!/usr/bin/env python3
"""FSB1 (float32) -> FSB2 (uint16): half the size for whole-number count tables (k-mer counts), nothing lost.

Refuses a table with any value that is not a whole number in 0..65535 (embeddings, transformed values).
Streams in blocks of rows, so memory stays small. fastsne reads both layouts.

usage: fsb_u16.py IN.fsb OUT.fsb      (OUT may not be IN)
"""
import os, struct, sys
import numpy as np

src, dst = sys.argv[1:3]
if os.path.abspath(src) == os.path.abspath(dst): sys.exit("OUT must differ from IN")
with open(src, "rb") as f:
    mg = f.read(4)
    if mg != b"FSB1": sys.exit(f"{src}: not an FSB1 file ({mg!r})")
    D, N = struct.unpack("<iq", f.read(12))
    tmp = dst + ".tmp"
    with open(tmp, "wb") as o:
        o.write(b"FSB2" + struct.pack("<iq", D, N))
        B = max(1, (64 << 20) // (4 * D))
        for s in range(0, N, B):
            m = min(B, N - s)
            X = np.frombuffer(f.read(4 * D * m), np.float32)
            if X.size != m * D: sys.exit(f"{src}: truncated")
            H = X.astype(np.uint16)
            if not np.array_equal(H, X): sys.exit(f"{src}: values that are not whole numbers in 0..65535 (row block {s}); not converted")
            o.write(H.tobytes())
        o.write(f.read())   # ids
os.replace(tmp, dst)
print(f"{dst}: {N} rows x {D}, {os.path.getsize(src) / 1e9:.2f} -> {os.path.getsize(dst) / 1e9:.2f} GB")
