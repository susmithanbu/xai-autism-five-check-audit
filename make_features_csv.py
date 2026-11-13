"""
Build features.csv for case4_gait_audit.py from the Al-Jubouri Kinect v2 gait dataset.

What it does
  1. Reads 'Final dataset.xlsx' (800 rows x 1,259 features + class; no child IDs).
  2. Reads every child's 8 per-recording files (augmentation\\1..7\\Nfeatures.xlsx and
     video\\8features.xlsx) and matches each one to its row in the final dataset,
     so every row gets a child_id.
  3. Marks the original recording of each child (the one whose translated copies,
     +/-0.3 m in x, exist among its siblings).
  4. Groups children whose RGB or skeleton videos are byte-identical (MD5).
  5. Adds age and sex from readme.docx.
  6. Computes shoulder and elbow angles and the atypical-arm-posture rule
     (>= 2 of 5 arm markers outside the TD 2.5th-97.5th percentile range).

Usage (from C:\\audit):
    py -3.12 make_features_csv.py "C:\\audit\\data\\Dataset-2\\Dataset"
Output: features.csv and make_features_log.txt
"""
import hashlib
import os
import re
import sys
import zipfile
from collections import Counter

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

warnings.simplefilter("ignore")
ROOT = sys.argv[1] if len(sys.argv) > 1 else r"C:\audit\data\Dataset-2\Dataset"
GROUPS = [("ASD", os.path.join("Autism", "children with ASD"), "A", 1),
          ("TD", "Typical", "T", 0)]
FP_SCALARS = ["RomMidx", "RomMidy", "StrLe", "Velocity", "GaCT", "MaxStLe", "MaxStWi"]
log = open("make_features_log.txt", "w", encoding="utf-8")


def say(*a):
    print(*a)
    print(*a, file=log)


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def joint_triples(df):
    """Map joint name -> (x, y, z) arrays from a per-recording features sheet."""
    cols = list(df.columns)
    out = {}
    for i, c in enumerate(cols[:-2]):
        c = str(c)
        if c.startswith("Unnamed") or c.startswith("H:M") or c.startswith("MidT"):
            continue
        if str(cols[i + 1]).startswith("Unnamed") and str(cols[i + 2]).startswith("Unnamed"):
            xyz = df.iloc[:, i:i + 3].apply(pd.to_numeric, errors="coerce").dropna().values
            out[c] = xyz
    return out


def first_value(df, col):
    if col not in df:
        return np.nan
    v = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(v.iloc[0]) if len(v) else np.nan


def find_joint(joints, *names):
    norm = {re.sub(r"[^a-z]", "", k.lower()): k for k in joints}
    for n in names:
        if n in norm:
            return joints[norm[n]]
    return None


def angle(a, b, c):
    """Angle at b (degrees) between b->a and b->c, per frame."""
    n = min(len(a), len(b), len(c))
    u, v = a[:n] - b[:n], c[:n] - b[:n]
    cos = (u * v).sum(1) / np.clip(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-9, None)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def arm_markers(joints):
    ss = find_joint(joints, "spineshoulder", "spineshoulders", "neck")
    out = {}
    for side, s in (("R", "right"), ("L", "left")):
        sh = find_joint(joints, f"shoulder{s}", f"{s}shoulder")
        el = find_joint(joints, f"elbow{s}", f"{s}elbow")
        wr = find_joint(joints, f"wrist{s}", f"{s}wrist")
        if ss is None or sh is None or el is None or wr is None:
            return None
        sa, ea = angle(ss, sh, el), angle(sh, el, wr)
        out[f"shoulder{side}_max"] = sa.max()
        out[f"shoulder{side}_mean"] = sa.mean()
        out[f"elbow{side}_mean"] = ea.mean()
    return out


def readme_metadata():
    for f in os.listdir(ROOT):
        if f.lower().endswith(".docx"):
            xml = zipfile.ZipFile(os.path.join(ROOT, f)).read("word/document.xml").decode("utf8", "ignore")
            text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", xml.replace("</w:p>", " ")))
            i_td = text.find("Metadata of Typical")
            i_asd = text.find("Metadata of Participants with ASD")
            meta = {}
            for grp, seg in (("TD", text[i_td:i_asd]), ("ASD", text[i_asd:])):
                for m in re.finditer(r"Case\s*(\d+)\s+([MF])\s+([\d.]+)", seg):
                    meta[(grp, int(m.group(1)))] = (m.group(2), float(m.group(3)))
            return meta
    return {}


# ---------------------------------------------------------------- 1. final dataset
final = pd.read_excel(os.path.join(ROOT, "Final dataset.xlsx")).copy()
say(f"Final dataset: {final.shape}, class counts {final['class'].value_counts().to_dict()}")
fp_final_cols = ["mean-x-Midspain", "mean-y-Midspain", "mean-z-Midspain"] + \
                [c for c in FP_SCALARS if c in final.columns]

# ---------------------------------------------------------------- 2. per-recording files
recs = []
for grp, sub, cls, lab in GROUPS:
    gdir = os.path.join(ROOT, sub)
    kids = sorted([d for d in os.listdir(gdir) if d.isdigit()], key=int)
    for k in kids:
        files = [(i, os.path.join(gdir, k, "augmentation", str(i), f"{i}features.xlsx")) for i in range(1, 8)]
        files.append((8, os.path.join(gdir, k, "video", "8features.xlsx")))
        for i, p in files:
            if not os.path.exists(p):
                say("MISSING", p)
                continue
            d = pd.read_excel(p)
            j = joint_triples(d)
            mid = find_joint(j, "midspain", "spinemid")
            fp = list(mid.mean(0)) if mid is not None else [np.nan] * 3
            fp += [first_value(d, c) for c in fp_final_cols[3:]]
            recs.append(dict(group=grp, cls=cls, label=lab, child=int(k), aug=i, path=p, fp=fp,
                             arm=arm_markers(j), joints=sorted(j)))
        print(f"\r{grp} child {k} read", end="")
print()
say(f"Read {len(recs)} recording files")
say("Joint names found in a file:", recs[0]["joints"])

# ---------------------------------------------------------------- 3. match files to final rows
final["child_id"] = None
final["aug"] = np.nan
fpF = final[fp_final_cols].apply(pd.to_numeric, errors="coerce").values
scale = np.nanstd(fpF, 0)
scale[scale == 0] = 1
worst = []
for grp, sub, cls, lab in GROUPS:
    rows = np.where(final["class"].astype(str).str.strip() == cls)[0]
    R = [r for r in recs if r["cls"] == cls]
    A = np.array([r["fp"] for r in R]) / scale
    B = fpF[rows] / scale
    cost = np.sqrt(np.nansum((A[:, None, :] - B[None, :, :]) ** 2, 2))
    ri, ci = linear_sum_assignment(cost)
    for a, b in zip(ri, ci):
        final.loc[rows[b], "child_id"] = f"{grp}_{R[a]['child']:02d}"
        final.loc[rows[b], "aug"] = R[a]["aug"]
        R[a]["row"] = rows[b]
    worst.append(cost[ri, ci].max())
    say(f"{grp}: matched {len(ri)} files to {len(rows)} rows; median distance "
        f"{np.median(cost[ri, ci]):.2e}, max {cost[ri, ci].max():.2e}")
if max(worst) > 1e-3:
    say("WARNING: some files did not match a final-dataset row exactly. Send me make_features_log.txt.")

# block-structure check: are each child's 8 rows contiguous?
blocks = final.groupby("child_id").apply(lambda g: g.index.max() - g.index.min() + 1)
say(f"Rows per child: {final['child_id'].value_counts().value_counts().to_dict()}; "
    f"children whose rows are contiguous: {(blocks == 8).sum()} of {len(blocks)}")

# ---------------------------------------------------------------- 4. original recording
final["is_original"] = 0
orig_aug = Counter()
for cid, g in final.groupby("child_id"):
    mx = g["mean-x-Midspain"].astype(float)
    cands = [idx for idx, v in mx.items()
             if any(abs(mx - (v + 0.3)) < 1e-6) and any(abs(mx - (v - 0.3)) < 1e-6)]
    # several recordings can share the base x (e.g. a y-only transform); prefer the lowest file number
    best = min(cands, key=lambda i: g.loc[i, "aug"]) if cands else None
    if best is None:
        best = g.index[g["aug"] == 1][0] if (g["aug"] == 1).any() else g.index[0]
        say(f"  {cid}: translation pattern not found, using augmentation 1 as original")
    final.loc[best, "is_original"] = 1
    orig_aug[int(final.loc[best, "aug"])] += 1
say(f"Original recording identified as file number: {dict(orig_aug)}")

# ---------------------------------------------------------------- 5. duplicate groups (MD5)
parent = {}


def find(x):
    while parent.setdefault(x, x) != x:
        x = parent[x]
    return x


hashes = {}
for grp, sub, cls, lab in GROUPS:
    gdir = os.path.join(ROOT, sub)
    for k in sorted([d for d in os.listdir(gdir) if d.isdigit()], key=int):
        cid = f"{grp}_{int(k):02d}"
        find(cid)
        for f in ("video.avi", "Svideo.avi"):
            p = os.path.join(gdir, k, "video", f)
            if os.path.exists(p):
                hashes.setdefault((f, md5(p)), []).append(cid)
dups = [v for v in hashes.values() if len(v) > 1]
for v in dups:
    for c in v[1:]:
        parent[find(c)] = find(v[0])
    say(f"Duplicate video set: {v}")
final["group_id"] = final["child_id"].map(find)
say(f"{len(dups)} duplicate sets; {final['group_id'].nunique()} independent groups")

# ---------------------------------------------------------------- 6. age and sex
meta = readme_metadata()
final["age"] = final["child_id"].map(lambda c: meta.get((c[:-3].rstrip("_"), int(c[-2:])), (None, np.nan))[1])
final["sex"] = final["child_id"].map(lambda c: {"M": 1, "F": 0}.get(meta.get((c[:-3].rstrip("_"), int(c[-2:])), ("?",))[0], np.nan))
o = final[final["is_original"] == 1]
say(f"Age/sex found for {o['age'].notna().sum()} of {len(o)} children")
say(o.groupby(o["child_id"].str[:2])[["age", "sex"]].agg(["mean", "count"]).to_string())

# ---------------------------------------------------------------- 7. atypical arm posture
arm = {}
for r in recs:
    if "row" in r and final.loc[r["row"], "is_original"] == 1 and r["arm"]:
        arm[final.loc[r["row"], "child_id"]] = r["arm"]
if arm:
    A = pd.DataFrame(arm).T
    td = A[A.index.str.startswith("TD")]
    out_hi = lambda c: A[c] > td[c].quantile(0.975)
    out_lo = lambda c: A[c] < td[c].quantile(0.025)
    hits = (out_hi("shoulderR_max").astype(int) + out_hi("shoulderR_mean") + out_hi("shoulderL_mean")
            + out_lo("elbowR_mean") + out_lo("elbowL_mean"))
    flag = (hits >= 2).astype(int)
    final["atypical_arm"] = final["child_id"].map(flag).fillna(0).astype(int)
    say(f"Atypical arm posture: ASD {flag[flag.index.str.startswith('ASD')].sum()} of "
        f"{flag.index.str.startswith('ASD').sum()}, TD {flag[flag.index.str.startswith('TD')].sum()} of "
        f"{flag.index.str.startswith('TD').sum()} (gait paper, duplicates removed: 24/49 and 3/43)")
    A.round(2).to_csv("arm_angles_per_child.csv")
else:
    say("Arm angles not computed: shoulder/elbow/wrist/spine-shoulder joints not found. "
        "The audit will skip the subgroup test.")

# ---------------------------------------------------------------- write
final["label"] = (final["class"].astype(str).str.strip() == "A").astype(int)
meta_cols = ["child_id", "group_id", "label", "is_original", "age", "sex", "aug"] + \
            (["atypical_arm"] if "atypical_arm" in final else [])
feat_cols = [c for c in final.columns if c not in meta_cols + ["class"]]
final[meta_cols + feat_cols].to_csv("features.csv", index=False)
say(f"\nDONE - features.csv written: {len(final)} rows, {len(feat_cols)} features")
log.close()
