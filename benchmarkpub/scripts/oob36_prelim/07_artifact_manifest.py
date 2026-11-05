#!/usr/bin/env python3
"""07_artifact_manifest.py — ARTIFACT_MANIFEST.csv for ch4-oob36-download-prelim-20260930.

Walks $DATA, $OUT and $CODE and records path + sha256 + bytes for every file
created by this task.  Nothing outside those three directories is touched.

Output: $OUT/ARTIFACT_MANIFEST.csv
"""

from __future__ import annotations

import csv
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, OUT, CODE  # noqa: E402

ROOTS = (("data", DATA), ("analysis", OUT), ("code", CODE))
SKIP_SUFFIX = (".pyc",)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    manifest_path = os.path.join(OUT, "ARTIFACT_MANIFEST.csv")
    rows = []
    for label, root in ROOTS:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for fn in sorted(filenames):
                if fn.endswith(SKIP_SUFFIX):
                    continue
                p = os.path.join(dirpath, fn)
                if os.path.abspath(p) == os.path.abspath(manifest_path):
                    continue                      # do not hash the manifest into itself
                rows.append({"root": label,
                             "relpath": os.path.relpath(p, root),
                             "path": p,
                             "bytes": os.path.getsize(p),
                             "sha256": sha256(p)})
    with open(manifest_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["root", "relpath", "path", "bytes", "sha256"])
        w.writeheader()
        w.writerows(rows)

    by_root = {}
    for r in rows:
        by_root[r["root"]] = by_root.get(r["root"], 0) + 1
    print(f"ARTIFACT_MANIFEST.csv: {len(rows)} files")
    for k in ("data", "analysis", "code"):
        print(f"   {k:9s} {by_root.get(k, 0)}")


if __name__ == "__main__":
    main()
