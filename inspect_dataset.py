"""
Inspect the Kinect gait dataset and write a report (inspect_report.txt)
so the features.csv converter can be written for this exact layout.

Usage (from C:\\audit):
    py -3.12 inspect_dataset.py "C:\\audit\\data\\Dataset-2\\Dataset"
"""
import os
import re
import sys
import zipfile
from collections import Counter

import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else r"C:\audit\data\Dataset-2\Dataset"
out = open("inspect_report.txt", "w", encoding="utf-8")


def w(*a):
    print(*a)
    print(*a, file=out)


w("ROOT:", root)
w("\n=== TOP-LEVEL TREE (2 levels) ===")
for name in sorted(os.listdir(root)):
    p = os.path.join(root, name)
    w(("[D] " if os.path.isdir(p) else "[F] ") + name)
    if os.path.isdir(p):
        for sub in sorted(os.listdir(p))[:8]:
            w("     " + sub)

# Group folders: any directory that contains numbered child folders
w("\n=== CHILD FOLDERS ===")
child_dirs = []
for dp, dns, fns in os.walk(root):
    nums = [d for d in dns if re.fullmatch(r"\d+", d)]
    if len(nums) >= 5:
        w(f"{dp}: {len(nums)} numbered child folders")
        child_dirs.append(dp)

for cd in child_dirs:
    first = os.path.join(cd, sorted([d for d in os.listdir(cd) if d.isdigit()], key=int)[0])
    w(f"\n--- full listing of {first} ---")
    for dp, dns, fns in os.walk(first):
        rel = os.path.relpath(dp, first)
        w(f"  [{rel}]  {len(fns)} files")
        for f in sorted(fns)[:15]:
            w(f"     {f}  ({os.path.getsize(os.path.join(dp, f))} bytes)")
    ext = Counter()
    for dp, dns, fns in os.walk(cd):
        ext.update(os.path.splitext(f)[1].lower() for f in fns)
    w(f"  file types under {cd}: {dict(ext)}")


def show_table(path):
    w(f"\n=== TABLE {path} ===")
    try:
        if path.lower().endswith((".xlsx", ".xls")):
            x = pd.ExcelFile(path)
            w("sheets:", x.sheet_names)
            sheets = x.sheet_names[:3]
            frames = [(s, x.parse(s)) for s in sheets]
        else:
            frames = [("csv", pd.read_csv(path))]
        for s, d in frames:
            w(f"[{s}] shape={d.shape}")
            cols = list(d.columns)
            w("first 12 cols:", cols[:12])
            w("last 20 cols:", cols[-20:])
            obj = [c for c in cols if not pd.api.types.is_numeric_dtype(d[c])]
            w("text columns:", obj[:20])
            for c in cols:
                if re.search(r"id|name|child|subject|age|sex|gender|class|label|case|file", str(c), re.I):
                    w(f"  {c}: {d[c].nunique()} unique, e.g. {d[c].dropna().astype(str).unique()[:6].tolist()}")
            w(d.iloc[:10, :4].to_string())
    except Exception as e:
        w("ERROR reading:", e)


w("\n=== EXCEL/CSV FILES (largest 12 shown in full) ===")
tables = []
for dp, dns, fns in os.walk(root):
    for f in fns:
        if f.lower().endswith((".xlsx", ".xls", ".csv")):
            tables.append(os.path.join(dp, f))
w(f"{len(tables)} table files in total")
tables.sort(key=lambda p: -os.path.getsize(p))
for p in tables[:12]:
    w(f"  {os.path.getsize(p):>12,}  {p}")
# Read the top-level ones plus one per-child example
read = [p for p in tables if os.path.dirname(p) == root or os.path.dirname(os.path.dirname(p)) == root][:4]
per_child = [p for p in tables if p not in read][:2]
for p in read + per_child:
    show_table(p)

w("\n=== README ===")
for f in os.listdir(root):
    if f.lower().endswith(".docx"):
        try:
            xml = zipfile.ZipFile(os.path.join(root, f)).read("word/document.xml").decode("utf8", "ignore")
            text = re.sub(r"<[^>]+>", " ", xml.replace("</w:p>", "\n"))
            w(re.sub(r"[ \t]+", " ", text)[:6000])
        except Exception as e:
            w("ERROR:", e)

out.close()
print("\nDONE - report saved to inspect_report.txt")
