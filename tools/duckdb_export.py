#!/usr/bin/env python3
"""Export TNFs (+ kraken taxon) from danaSeq dana.duckdb files into fastsne binary input.

    duckdb_export.py OUTDIR db1.duckdb [db2.duckdb ...]   (or: duckdb_export.py OUTDIR @list.txt)

Per database writes OUTDIR/<flowcell>__<barcode>.fsb and .meta.tsv (the stats table holds contig, not read,
lengths, so no length is exported):
  .fsb       b"FSB1", int32 D, int64 N, N*D float32 (row-major), then N ids joined by "\\n"
  .meta.tsv  id, taxon
Skips databases whose outputs are newer than the database. Needs numpy + duckdb (e.g. the dana-tools env).
"""
import os
import struct
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import duckdb
import numpy as np


def out_name(db):
    parts = os.path.normpath(db).split(os.sep)
    return f"{parts[-3]}__{parts[-2]}"


def export(db, outdir):
    base = os.path.join(outdir, out_name(db))
    fsb, meta = base + ".fsb", base + ".meta.tsv"
    if os.path.exists(fsb) and os.path.exists(meta) and os.path.getmtime(fsb) >= os.path.getmtime(db):
        return db, -1, "up to date"
    try:
        con = duckdb.connect(db, read_only=True)
    except Exception as e:  # locked by a writer, corrupt, ...
        return db, 0, f"open failed: {e}"
    tables = {r[0] for r in con.execute("select table_name from information_schema.tables").fetchall()}
    if "tetra_data" not in tables:
        return db, 0, "no tetra_data"
    cols = [r[0] for r in con.execute("describe tetra_data").fetchall()][1:]
    sel = ", ".join(f't."{c}"' for c in cols)
    taxon = "k.taxa_name" if "kraken" in tables else "NULL"
    joins = (" left join (select seqid, any_value(taxa_name) taxa_name from kraken group by seqid) k using(seqid)"
             if "kraken" in tables else "")
    res = con.execute(f"select t.seqid, {taxon}, {sel} from tetra_data t{joins}").fetchnumpy()
    con.close()
    ids = res["seqid"]
    n, d = len(ids), len(cols)
    X = np.empty((n, d), np.float32)
    for j, c in enumerate(cols):
        X[:, j] = res[c]
    tmp = fsb + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b"FSB1")
        f.write(struct.pack("<iq", d, n))
        X.tofile(f)
        f.write("\n".join(map(str, ids)).encode())
    os.replace(tmp, fsb)
    tax = res[list(res.keys())[1]]
    with open(meta + ".tmp", "w") as f:
        f.write("id\ttaxon\n")
        for i in range(n):
            t = tax[i]
            f.write(f"{ids[i]}\t{'' if t is None or t is np.ma.masked else t}\n")
    os.replace(meta + ".tmp", meta)
    return db, n, "ok"


def main():
    outdir = sys.argv[1]
    dbs = []
    for a in sys.argv[2:]:
        if a.startswith("@"):
            dbs += [l.strip() for l in open(a[1:]) if l.strip()]
        else:
            dbs.append(a)
    os.makedirs(outdir, exist_ok=True)
    workers = int(os.environ.get("WORKERS", "4"))
    total = done = 0
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(export, db, outdir) for db in dbs]
        for fu in as_completed(futs):
            db, n, msg = fu.result()
            done += 1
            total += max(n, 0)
            print(f"[{done}/{len(dbs)}] {db}\t{n}\t{msg}\t(total {total})", flush=True)


if __name__ == "__main__":
    main()
