"""
Case 4 (positive control): five-check explanation audit of the Kinect v2 gait model.

Produces every number marked [TO FILL] in the Case 4 parts of the manuscript.

INPUT: one CSV with one row per recording (original + augmentations), containing
    child_id      child identifier
    group_id      duplicate-set identifier (children sharing byte-identical files
                  get the same group_id; otherwise equal to child_id)
    label         0 = typically developing, 1 = ASD
    is_original   1 for the original recording, 0 for an augmented copy
    age, sex      age in years; sex coded 0/1
    atypical_arm  (optional) 1 if the child met the atypical-arm-posture rule
    ... all engineered feature columns (the 1,265 Kinect features)

Edit the CONFIG block below so the feature-name patterns match your columns.

Usage:
    pip install scikit-learn shap pandas scipy statsmodels
    python case4_gait_audit.py features.csv --n-perm 100 --out case4_results.json
    (use --n-perm 20 for a quick first run)
"""

import argparse
import json
import re
import sys

import numpy as np
import pandas as pd
import shap
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

# ----------------------------------------------------------------------------
# CONFIG - adapt to your column names
# ----------------------------------------------------------------------------
META_COLS = ["child_id", "group_id", "label", "is_original", "age", "sex", "atypical_arm", "aug"]

# Features that depend on walking distance or camera position (removed from the
# main model, used alone as the "recording geometry" shortcut baseline).
# Set for the Al-Jubouri dataset: absolute joint positions (mean-x/y/z-<joint>),
# every depth-axis (z) feature, gait timing/speed/stride, and hand-tip positions.
GEOMETRY_PATTERNS = [
    r"^mean-[xyz]-",                       # absolute joint positions
    r"-z-",                                # depth-axis variance/std
    r"^(MaxStLe|MaxStWi|StrLe|GaCT|StaT|SwiT|Velocity|HaTiLPos|HaTiRPos|MaxDBFE|MinDBFE|Threshold)$",
    r"^Rom\w+z$",                          # depth-axis range of motion, if present
]

# Abbreviations used in this dataset's distance (MidT..) and range-of-motion (Rom..) columns
ABBR = {"An": "lower_limb", "Fo": "lower_limb", "Ke": "lower_limb", "Hi": "lower_limb",
        "El": "upper_limb", "Ha": "upper_limb", "HaT": "upper_limb", "Th": "upper_limb",
        "Wr": "upper_limb", "Sh": "upper_limb",
        "H": "trunk_head", "Ne": "trunk_head", "Sb": "trunk_head", "SSh": "trunk_head",
        "Mid": "trunk_head", "Sp": "trunk_head",
        "Ei": "upper_limb", "Ssh": "trunk_head", "sb": "trunk_head", "Ank": "lower_limb",
        "HTi": "upper_limb", "Gr": None}   # Gr = ground (floor); takes the other joint's region

# Joint angles, named by their two outer joints; region = the joint at the vertex
ANGLES = {"HESH": "trunk_head",   # head-shoulder (neck-shoulder angle)
          "SPEL": "upper_limb",   # spine-elbow  -> shoulder angle
          "SHWR": "upper_limb",   # shoulder-wrist -> elbow angle
          "ELHA": "upper_limb",   # elbow-hand -> wrist angle
          "THHA": "upper_limb", "THHTI": "upper_limb",
          "SPKN": "lower_limb",   # spine-knee -> hip angle
          "HIAN": "lower_limb",   # hip-ankle -> knee angle
          "KNFO": "lower_limb"}   # knee-foot -> ankle angle
_TOK = r"(?:HaT|HTi|SSh|Ssh|Ank|Mid|Ha|An|El|Ei|Fo|Hi|Ke|Ne|Sh|Sb|sb|Th|Wr|Gr|H)"
PAIR_RE = re.compile(rf"^(?:mean|variance|std)\s+({_TOK})[LR]?T({_TOK})[LR]?$")

# Body-region domains. Tokens are matched after lower-casing and removing
# non-letters, so "ShoulderLeft", "left_shoulder" and "Left Shoulder" all match.
# "spineshoulder" is a trunk joint and is checked before "shoulder".
TRUNK_TOKENS = ["spineshoulder", "spinemid", "spinebase", "spine", "neck", "head", "trunk"]
UPPER_TOKENS = ["shoulder", "elbow", "wrist", "handtip", "hand", "thumb", "arm"]
LOWER_TOKENS = ["hip", "knee", "ankle", "foot", "shank", "thigh", "leg"]

N_TREES = 400
N_FOLDS = 5
SEEDS = (0, 1, 2)          # check 3: retraining stability
NOISE_SD = 0.05            # check 3: input noise, in training-SD units
TOP_K = (1, 2, 3)          # check 2: number of domains removed
N_RANDOM_ORDERS = 20       # check 2: random-removal baseline draws
# ----------------------------------------------------------------------------


def region_of(col):
    c = col.strip()
    a = re.match(r"^(?:mean|variance|std)\s+([A-Z]+)[LR]$", c)
    if a and a.group(1) in ANGLES:
        return ANGLES[a.group(1)]
    p = PAIR_RE.match(c)
    if p and not c.split()[-1].startswith("MidT"):
        r = {ABBR[p.group(1)], ABBR[p.group(2)]} - {None}
        return r.pop() if len(r) == 1 else "whole_body"
    m = re.match(r"^(?:mean|variance|std)?\s*MidT([A-Za-z]+)$", c) or \
        re.match(r"^Rom([A-Za-z]+?)[LR]?[xyz]$", c)
    if m:
        a = m.group(1)
        if a in ABBR:
            return ABBR[a]
        if len(a) > 1 and a[-1] in "LR" and a[:-1] in ABBR:
            return ABBR[a[:-1]]
    s = re.sub(r"[^a-z]", "", col.lower())
    s = s.replace("midspain", "spine")
    found = set()
    for tok in TRUNK_TOKENS:
        if tok in s:
            found.add("trunk_head")
            s = s.replace(tok, "")
    for tok in UPPER_TOKENS:
        if tok in s:
            found.add("upper_limb")
            s = s.replace(tok, "")
    for tok in LOWER_TOKENS:
        if tok in s:
            found.add("lower_limb")
            s = s.replace(tok, "")
    if len(found) == 1:
        return found.pop()
    if len(found) > 1:
        return "whole_body"
    return "unassigned"


def is_geometry(col):
    return any(re.search(p, col, flags=re.I) for p in GEOMETRY_PATTERNS)


def child_level_permutation(df, rng):
    """Shuffle labels between groups (children), keeping all copies of a child together."""
    g = df.groupby("group_id")["label"].first()
    perm = pd.Series(rng.permutation(g.values), index=g.index)
    return df["group_id"].map(perm).values


def positive_class_shap(explainer, X):
    sv = explainer.shap_values(X, check_additivity=False)
    if isinstance(sv, list):          # older shap: list per class
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # newer shap: (n, features, classes)
        return sv[:, :, 1]
    return sv


def domain_shares(sv, domain_idx):
    a = np.abs(sv)
    s = np.stack([a[:, idx].sum(1) for idx in domain_idx], 1)
    return s / np.clip(s.sum(1, keepdims=True), 1e-12, None)


def run_cv(df, X, y, seed, explain=False, domain_idx=None, noise=False):
    """Child-level CV: train on all rows of training children, test on originals of unseen children."""
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)  # fixed split
    orig = df["is_original"].values == 1
    groups = df["group_id"].values
    n = len(df)
    proba = np.full(n, np.nan)
    shares = np.full((n, len(domain_idx) if domain_idx else 0), np.nan)
    feat_sv = np.full((n, X.shape[1]), np.nan) if explain else None
    shares_noise = np.full_like(shares, np.nan)
    faith = {"top": {k: [] for k in TOP_K}, "rand": {k: [] for k in TOP_K}}
    rng = np.random.default_rng(seed + 100)
    for tr, te in cv.split(X, y, groups):
        te = te[orig[te]]
        m = RandomForestClassifier(n_estimators=N_TREES, random_state=seed, n_jobs=-1)
        m.fit(X[tr], y[tr])
        proba[te] = m.predict_proba(X[te])[:, 1]
        if not explain:
            continue
        ex = shap.TreeExplainer(m)
        sv = positive_class_shap(ex, X[te])
        feat_sv[te] = sv
        sh = domain_shares(sv, domain_idx)
        shares[te] = sh
        if noise:
            sd = X[tr].std(0)
            Xn = X[te] + rng.normal(0, NOISE_SD, X[te].shape) * sd
            shares_noise[te] = domain_shares(positive_class_shap(ex, Xn), domain_idx)
        # Check 2: faithfulness by domain removal (reset to training median)
        med = np.median(X[tr], 0)
        P = m.predict_proba(X[te])
        c = P.argmax(1)
        base = P[np.arange(len(te)), c]
        order = np.argsort(-sh, 1)
        for k in TOP_K:
            Xd = X[te].copy()
            for i in range(len(te)):
                for d in order[i, :k]:
                    Xd[i, domain_idx[d]] = med[domain_idx[d]]
            faith["top"][k].extend(base - m.predict_proba(Xd)[np.arange(len(te)), c])
            drops = []
            for _ in range(N_RANDOM_ORDERS):
                Xr = X[te].copy()
                for i in range(len(te)):
                    for d in rng.permutation(len(domain_idx))[:k]:
                        Xr[i, domain_idx[d]] = med[domain_idx[d]]
                drops.append(base - m.predict_proba(Xr)[np.arange(len(te)), c])
            faith["rand"][k].extend(np.mean(drops, 0))
    return dict(proba=proba, shares=shares, shares_noise=shares_noise,
                feat_sv=feat_sv, faith=faith)


def scores(y, proba, mask):
    return (balanced_accuracy_score(y[mask], proba[mask] >= 0.5),
            roc_auc_score(y[mask], proba[mask]))


def mean_participant_rho(A, B):
    r = [spearmanr(a, b)[0] for a, b in zip(A, B)]
    return float(np.nanmean(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--out", default="case4_results.json")
    ap.add_argument("--age-min", type=float, default=None, help="keep children aged >= this")
    ap.add_argument("--age-max", type=float, default=None, help="keep children aged <= this")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.age_min is not None or args.age_max is not None:
        lo = -np.inf if args.age_min is None else args.age_min
        hi = np.inf if args.age_max is None else args.age_max
        df = df[(df["age"] >= lo) & (df["age"] <= hi)]
        kids = df[df["is_original"] == 1]
        print(f"Age-matched subset {lo}-{hi} years: {len(kids)} children "
              f"(ASD {int((kids['label'] == 1).sum())}, TD {int((kids['label'] == 0).sum())}); "
              f"mean age ASD {kids.loc[kids['label'] == 1, 'age'].mean():.2f}, "
              f"TD {kids.loc[kids['label'] == 0, 'age'].mean():.2f}")
    for c in ["child_id", "label", "is_original"]:
        if c not in df:
            sys.exit(f"missing column: {c}")
    if "group_id" not in df:
        df["group_id"] = df["child_id"]
    df = df.reset_index(drop=True)
    y = df["label"].astype(int).values
    orig = df["is_original"].values == 1

    feat_cols = [c for c in df.columns if c not in META_COLS]
    geo_cols = [c for c in feat_cols if is_geometry(c)]
    main_cols = [c for c in feat_cols if c not in geo_cols]
    regions = {c: region_of(c) for c in main_cols}
    names = ["upper_limb", "lower_limb", "trunk_head", "whole_body"]
    unassigned = [c for c, r in regions.items() if r == "unassigned"]
    if unassigned:
        names.append("unassigned")
        print(f"WARNING: {len(unassigned)} features not assigned to a body region "
              f"(first: {unassigned[:5]}). Edit the token lists or accept an 'unassigned' domain.")
    domain_idx = [np.array([i for i, c in enumerate(main_cols) if regions[c] == d]) for d in names]
    names_used = [d for d, idx in zip(names, domain_idx) if len(idx)]
    domain_idx = [idx for idx in domain_idx if len(idx)]
    print(f"{len(feat_cols)} features: {len(main_cols)} main, {len(geo_cols)} geometry-dependent")
    print("domain sizes:", {d: len(i) for d, i in zip(names_used, domain_idx)})

    X = df[main_cols].values.astype(float)
    res = {"n_children_test": int(orig.sum()), "domains": names_used,
           "domain_sizes": [int(len(i)) for i in domain_idx]}

    # ---------------- Check 1: predictive validity ----------------
    real = run_cv(df, X, y, seed=0, explain=True, domain_idx=domain_idx, noise=True)
    ba, auc = scores(y, real["proba"], orig)
    rng = np.random.default_rng(0)
    null = []
    for i in range(args.n_perm):
        yp = child_level_permutation(df, rng)
        p = run_cv(df, X, yp, seed=0)["proba"]
        null.append(balanced_accuracy_score(yp[orig], p[orig] >= 0.5))
        print(f"\rpermutation {i + 1}/{args.n_perm}", end="")
    print()
    res["check1"] = {"BA": ba, "AUC": auc, "perm_chance": float(np.mean(null)),
                     "perm_p": float((1 + sum(v >= ba for v in null)) / (args.n_perm + 1))}

    # Age-and-sex baseline, same child-level splits and its own permutation test
    if {"age", "sex"} <= set(df.columns):
        Xas = df[["age", "sex"]].values.astype(float)
        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)

        def as_ba(lbl):
            pr = np.full(len(df), np.nan)
            for tr, te in cv.split(Xas, lbl, df["group_id"]):
                te = te[orig[te]]
                m = make_pipeline(StandardScaler(), LogisticRegression()).fit(Xas[tr], lbl[tr])
                pr[te] = m.predict_proba(Xas[te])[:, 1]
            return pr
        pr = as_ba(y)
        b_as, a_as = scores(y, pr, orig)
        rng2 = np.random.default_rng(1)
        null_as = []
        for _ in range(args.n_perm):
            yp = child_level_permutation(df, rng2)
            null_as.append(balanced_accuracy_score(yp[orig], as_ba(yp)[orig] >= 0.5))
        res["baseline_age_sex"] = {"BA": b_as, "AUC": a_as, "perm_chance": float(np.mean(null_as)),
                                   "perm_p": float((1 + sum(v >= b_as for v in null_as)) / (args.n_perm + 1))}

    if geo_cols:
        pg = run_cv(df, df[geo_cols].values.astype(float), y, seed=0)["proba"]
        res["baseline_geometry_only"] = dict(zip(["BA", "AUC"], scores(y, pg, orig)))

    # Leaky record-level split, for contrast
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=0)
    pl = np.full(len(df), np.nan)
    for tr, te in skf.split(X, y):
        pl[te] = RandomForestClassifier(N_TREES, random_state=0, n_jobs=-1).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    res["record_level_leaky"] = dict(zip(["BA", "AUC"], scores(y, pl, np.ones(len(df), bool))))

    # ---------------- Check 2: faithfulness ----------------
    f = real["faith"]
    res["check2"] = {
        "AOPC_shap": float(np.mean([np.mean(f["top"][k]) for k in TOP_K])),
        "AOPC_random": float(np.mean([np.mean(f["rand"][k]) for k in TOP_K])),
        "drop_by_k_shap": {k: float(np.mean(f["top"][k])) for k in TOP_K},
        "drop_by_k_random": {k: float(np.mean(f["rand"][k])) for k in TOP_K},
    }

    # ---------------- Check 3: stability ----------------
    S = real["shares"][orig]
    res_noise = mean_participant_rho(S, real["shares_noise"][orig])
    runs = [real] + [run_cv(df, X, y, seed=s, explain=True, domain_idx=domain_idx) for s in SEEDS[1:]]
    pairs = [(a, b) for i, a in enumerate(runs) for b in runs[i + 1:]]
    grp = [spearmanr(a["shares"][orig].mean(0), b["shares"][orig].mean(0))[0] for a, b in pairs]
    part = [mean_participant_rho(a["shares"][orig], b["shares"][orig]) for a, b in pairs]
    feat_grp = [spearmanr(np.abs(a["feat_sv"][orig]).mean(0), np.abs(b["feat_sv"][orig]).mean(0))[0]
                for a, b in pairs]
    res["check3"] = {"noise_rho": res_noise, "seeds_group_rho": float(np.mean(grp)),
                     "seeds_participant_rho": float(np.mean(part)),
                     "seeds_feature_level_group_rho": float(np.mean(feat_grp))}

    # ---------------- Check 4: randomized-model sanity ----------------
    yp = child_level_permutation(df, np.random.default_rng(7))
    perm = run_cv(df, X, yp, seed=0, explain=True, domain_idx=domain_idx)
    res["check4"] = {
        "participant_rho": mean_participant_rho(S, perm["shares"][orig]),
        "group_rho": float(spearmanr(S.mean(0), perm["shares"][orig].mean(0))[0]),
        "feature_level_group_rho": float(spearmanr(np.abs(real["feat_sv"][orig]).mean(0),
                                                   np.abs(perm["feat_sv"][orig]).mean(0))[0]),
    }

    # ---------------- Check 5: plausibility and discrimination ----------------
    yo = y[orig]
    res["check5"] = {"mean_share": dict(zip(names_used, S.mean(0).round(4).tolist())),
                     "mean_share_TD": dict(zip(names_used, S[yo == 0].mean(0).round(4).tolist())),
                     "mean_share_ASD": dict(zip(names_used, S[yo == 1].mean(0).round(4).tolist()))}
    sizes = np.array([len(i) for i in domain_idx], float)
    per_feat = S.mean(0) / sizes
    res["check5"]["size_normalized_share"] = dict(zip(names_used, (per_feat / per_feat.sum()).round(4).tolist()))
    pvals = [mannwhitneyu(S[yo == 1, j], S[yo == 0, j]).pvalue for j in range(len(names_used))]
    res["check5"]["ASD_vs_TD_q"] = dict(zip(names_used, multipletests(pvals, method="fdr_bh")[1].round(4).tolist()))
    if "atypical_arm" in df and "upper_limb" in names_used:
        j = names_used.index("upper_limb")
        arm = df["atypical_arm"].values[orig]
        sub, rest = S[(yo == 1) & (arm == 1), j], S[(yo == 1) & (arm == 0), j]
        res["check5"]["upper_share_subgroup"] = float(sub.mean())
        res["check5"]["upper_share_other_ASD"] = float(rest.mean())
        res["check5"]["subgroup_p"] = float(mannwhitneyu(sub, rest, alternative="greater").pvalue)

    # ---------------- verdicts ----------------
    c = res
    v = {
        "1_validity": c["check1"]["perm_p"] < 0.05,
        "2_faithfulness": c["check2"]["AOPC_shap"] > 2 * c["check2"]["AOPC_random"],
        "3a_noise": c["check3"]["noise_rho"] >= 0.7,
        "3b_retraining": c["check3"]["seeds_participant_rho"] >= 0.7,
        "4_sanity": c["check4"]["participant_rho"] <= 0.2,
    }
    if "subgroup_p" in c["check5"]:
        v["5b_discrimination"] = c["check5"]["subgroup_p"] < 0.05
    res["verdicts"] = v
    res["release_gate"] = "Release" if all(v.values()) else "Withhold"

    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))
    print("\nNOTE: with only four domains a participant-level Spearman rho is coarse; "
          "report the feature-level rho beside it. 'AOPC clearly above random' is "
          "operationalised here as more than twice the random AOPC - state this in Methods.")


if __name__ == "__main__":
    main()
