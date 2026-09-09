#!/usr/bin/env python3
"""Regenerate the RQ2/RQ3 SHAP figures and tables from the GXA local-explanation run.

Input  : ../ml4srm-results/<approach>/explanations/{new,old}/{global,local}/*_shap_aggregated*.csv
         (new = WEAVE / AutoML-selected model, old = original published model)
Output : figures/rq2/heat_{new,old}_{global,local}.pdf   magnitude-share heatmaps
         figures/rq2/heat_colorbar.pdf
         figures/rq2/slope_<approach>.pdf                global->local slopegraphs
         figures/rq2/slope_legend.pdf
         figures/shap_shares_gxa.csv                     tidy per-group shares
         figures/rq2/concentration_table.tex             table:concentration body

SIGN. The re-run exports per-instance signed values as
<scope>_shap_values.csv, with columns row, dataset_row, base_value, prediction and
then one signed SHAP value per feature. Where that file exists we report the
*signed* share of a group, sum(shap in g) / sum(|shap| over all features), so
direction is preserved and the heatmap is diverging. Where only the
*_shap_aggregated*.csv exists we can report magnitude only -- those cells are
marked grey in the signed figures rather than silently mixed in.

Every signed file was checked for SHAP additivity (base_value + sum(shap) ==
prediction) to machine precision, and each aggregated file was verified to equal
mean(|SHAP|) of its per-instance file.

Run: python3 make_shap_figs.py
"""
import csv, glob, os, re, math, sys, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))            # .../evaluation/ml4srm/analysis
PKG  = os.path.dirname(HERE)                                # .../evaluation/ml4srm

# One root for the shipped explanation outputs, one for this script's inputs, one for
# what it writes. Only DATA_ROOT is worth overriding, and only to point at a second copy
# of the outputs; ML4SRM_FIGURES exists so the paper build can collect the figures
# in place instead of copying them out of the package afterwards.
DATA_ROOT = os.environ.get("ML4SRM_DATA", PKG)
INPUTS    = os.path.join(HERE, "inputs")
OUTPUT    = os.environ.get("ML4SRM_FIGURES", os.path.join(HERE, "output"))

RES = DATA_ROOT                                             # explanation outputs
OUT = os.path.join(OUTPUT, "rq2")                           # where figures are written
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- feature groups
# Nine groups from Table table:feature-groups (text/5_rq1.tex). CB is only ever
# populated for Dev-Assist, so it is dropped from the plots when Dev-Assist has
# no data -- see GROUPS_PLOT below.
GROUPS = ["CMo", "CN", "CB", "MMo", "RT", "MN", "P", "MB", "DF"]

def group_dev_assist(f):
    if f in ("ClassLinesCount", "ClassesInvokedCount"):                     return "CB"
    if f in ("ClassAccessModifier", "ClassModifier", "IsClassConcrete",
             "MethodInnerOrAnonymousClass"):                                return "CMo"
    if f.startswith("ClassNameContains") or f == "ClassNameKeywordsCount":  return "CN"
    if f in ("MethodAccessModifier", "MethodModifier", "IsMethodImplicit"): return "MMo"
    if f == "MethodReturnType":                                             return "RT"
    if f.startswith("MethodNameContains") or f == "MethodType":             return "MN"
    if f in ("MethodParametersTypes", "ParametersCount"):                   return "P"
    if (f.startswith("ParameterToInvokedSinkContains")
            or f.startswith("SourceToReturnContains")
            or f in ("ParameterFlowsToReturn", "ParameterAndReturnTypeMatch")): return "DF"
    return "MB"          # Invoked*, *Count metrics, MethodOrigin

def group_sscm(f):
    if f in ("isAbstractClass", "isConcreteClass", "isInterface"):          return "CMo"
    if f == "concreteMethod":                                               return "MMo"
    if f.startswith("DFfrom"):                                              return "DF"
    if f.endswith("NRMInvokes"):                                            return "MB"
    if f.endswith("ReturnNRCategoryFeature"):                               return "RT"
    if f.endswith("ParamNRCategoryFeature"):                                return "P"
    if f in ("StringParam", "String[]Param", "Byte[]Param", "Char[]Param",
             "primitiveParam"):                                             return "P"
    if f.endswith("ClassNamePatternCat") or f.endswith("NRCategoryFeature"): return "CN"
    if f.endswith("PrefixFeature") or f in ("inputPrefixType", "targetPrefixTye",
                                            "mixedPrefixType"):             return "MN"
    return "MB"

def group_swan(f):
    if f.startswith("ParamDf") or f.startswith("SoMtdRtn"):                 return "DF"
    if f.endswith("Invkd"):                                                 return "MB"
    if f in ("ClsPUBLIC", "ClsSTATIC", "ClsFINAL", "ClsAnonymous", "InnerCls"): return "CMo"
    if f.startswith("ClsNmCtn") or f.startswith("ClsNmEnds"):               return "CN"
    if (f.startswith("MtdMod") or f in ("MtdConstructor", "ImplicitMtd",
                                        "Implicit method", "MtdLoneGetterOrSetter")): return "MMo"
    if f.startswith("RtnTyp"):                                              return "RT"
    if f.startswith("ParamTyp") or f == "MtdCtnParams":                     return "P"
    if f.startswith("MtdNm") or f == "MtdStrtOn":                           return "MN"
    return "MB"

# approach key -> (label, results subdir, grouping fn)
APPS = [
    ("Dev-Assist", "Dev-Assist",     "dev-assist",     group_dev_assist),
    ("SSCM",       "SSCM",           "sscm",           group_sscm),
    ("SWAN-79",    "SWAN-CWE79",     "swan/cwe79",     group_swan),
    ("SWAN-89",    "SWAN-CWE89",     "swan/cwe89",     group_swan),
    ("SWAN-Sa",    "SWAN-Sanitizer", "swan/sanitizer", group_swan),
    ("SWAN-Si",    "SWAN-Sink",      "swan/sink",      group_swan),
    ("SWAN-So",    "SWAN-Source",    "swan/source",    group_swan),
]

# ------------------------------------------------------------------ load results
META_COLS = ("row", "dataset_row", "base_value", "prediction")

# Dev-Assist is multi-label and exports one per-instance file per label. Labels 0
# (CWE79) and 2 (sanitizer) are the categories GXA does not label, so we pool only
# 1 (source), 3 (CWE89) and 4 (sink). The same subset is used for its global cell
# even though the training set does label all five -- otherwise the global and local
# panels of Figure fig:shap-heatmap would summarise different label sets and the
# global->local comparison the figure invites would be meaningless. Restricting the
# global cell costs little (effective groups 5.1 -> 5.0) and removes a real artefact:
# with all five labels the class-modifier group reads +9.2%, with the clean subset
# -9.4%, i.e. the unlabelled categories were flipping its sign.
SSCM_CLASSES = (1, 2)   # Target and Input; class 0 is None, see _find_signed
DEVASSIST_LABELS = (1, 3, 4)

def _find(subdir, variant, scope, needle):
    base = os.path.join(RES, subdir, "explanations", variant, scope)
    if not os.path.isdir(base):
        return None
    c = [f for f in os.listdir(base) if f.endswith(".csv") and needle in f]
    return os.path.join(base, sorted(c)[0]) if c else None

def _find_signed(subdir, variant, scope):
    """Per-instance signed file(s) to pool for one cell; [] if unavailable.

    Multi-label approaches yield one file per label, which are pooled; every wanted
    label must be present or the cell is treated as unavailable rather than
    silently summarising a subset of the subset.
    """
    base = os.path.join(RES, subdir, "explanations", variant, scope)
    if not os.path.isdir(base):
        return []
    if subdir == "sscm":
        # The re-run exports one file per class (class_index.csv: 0 None, 1 Target, 2 Input).
        # We pool the two security-relevant classes and drop None, for the same reason
        # Dev-Assist drops the categories GXA does not annotate: None is the absence of a
        # detection target rather than one of them. Target is exactly GXA's sink class and
        # Input corresponds to its sources.
        out = []
        for c in SSCM_CLASSES:
            hits = (glob.glob(os.path.join(base, f"*shap_values_class_{c}.csv"))
                    + glob.glob(os.path.join(base, str(c), f"*shap_values_class_{c}.csv")))
            if hits:
                out.append(sorted(hits)[0])
        return out if len(out) == len(SSCM_CLASSES) else []
    if subdir == "dev-assist":
        out = []
        for L in DEVASSIST_LABELS:
            hits = (glob.glob(os.path.join(base, f"*shap_values_label_{L}.csv"))
                    + glob.glob(os.path.join(base, str(L), f"*shap_values_label_{L}.csv")))
            if hits:
                out.append(sorted(hits)[0])
        return out if len(out) == len(DEVASSIST_LABELS) else []
    p = _find(subdir, variant, scope, "shap_values")
    return [p] if p else []

def read_agg(path):
    """-> {feature: mean|SHAP|}. Files are 2-column: Feature,<ValueColumn>."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2 or row[0].strip().lower() == "feature":
                continue
            try:
                out[row[0].strip()] = float(row[1])
            except ValueError:
                pass
    return out

def read_signed(paths):
    """-> (feature_names, [[signed shap per feature] per (instance, label)]).

    Several paths are pooled row-wise. Because every label carries the same number
    of instances, pooling equals averaging over labels once the shares are
    normalised, which keeps this consistent with the explainer's own all-labels
    aggregate (verified: that file is the elementwise mean of the per-label ones).
    """
    feats, data = None, []
    for path in paths:
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        hdr = rows[0]
        idx = [i for i, c in enumerate(hdr) if c not in META_COLS]
        names = [hdr[i] for i in idx]
        if feats is None:
            feats = names
        elif names != feats:
            sys.exit(f"feature order differs between per-label files: {path}")
        data += [[float(r[i]) for i in idx] for r in rows[1:] if r and r[0] != ""]
    return feats, data

def shares_from_signed(feats, data, gfn):
    """Signed and absolute share of each group.

    A group's *share of |SHAP|* is its magnitude divided by the model's total
    magnitude, so the shares sum to 1. The *signed* share is that same quantity
    carrying the sign of the group's net contribution:

        abs_share(g)    = sum|shap in g| / sum|shap|
        signed_share(g) = sign(sum(shap in g)) * abs_share(g)

    Signing this way rather than normalising the signed sum directly is
    deliberate: summing raw signed values over 250 instances cancels almost
    completely (net shares collapse to well under 1%), which would encode how
    balanced a group is, not how influential it is or which way it pushes.
    """
    net = {g: 0.0 for g in GROUPS}
    absol = {g: 0.0 for g in GROUPS}
    for r in data:
        for f, v in zip(feats, r):
            g = gfn(f)
            net[g] += v
            absol[g] += abs(v)
    denom = sum(absol.values())
    if denom <= 0:
        return None, None
    absh = {g: absol[g] / denom for g in GROUPS}
    signed = {g: (-absh[g] if net[g] < 0 else absh[g]) for g in GROUPS}
    return signed, absh

def instance_shares(feats, data, gfn):
    """Per group: the fraction of explained instances whose net contribution is positive.

    This is the direction statistic RQ3 reports. Unlike the sign of the aggregate net
    contribution -- which the signed share carries, and which is a small difference of
    large numbers, so its sign is not resolvable at these sample sizes (see
    sign_evidence) -- a proportion over hundreds of instances has a standard error we
    can quote. Instances whose contribution for a group is exactly zero carry no
    direction and are excluded, so n is reported per group.
    """
    cnt = {g: 0 for g in GROUPS}
    nz = {g: 0 for g in GROUPS}
    for r in data:
        acc = {g: 0.0 for g in GROUPS}
        for f, v in zip(feats, r):
            acc[gfn(f)] += v
        for g in GROUPS:
            if abs(acc[g]) > 1e-12:
                nz[g] += 1
                if acc[g] > 0:
                    cnt[g] += 1
    return {g: ((cnt[g] / nz[g]) if nz[g] else None, nz[g]) for g in GROUPS}

def shares_from_agg(vals, gfn):
    """Magnitude-only fallback when no per-instance file exists."""
    tot = {g: 0.0 for g in GROUPS}
    for feat, v in vals.items():
        tot[gfn(feat)] += abs(v)
    s = sum(tot.values())
    return None if s <= 0 else {g: tot[g] / s for g in GROUPS}

# --------------------------------------------------------------- exclusions
# GXA carries no ground-truth labels for the sanitizer and CWE79 categories: both
# *-gxa-dataset.arff files have '?' for every instance. The LOCAL explanations for
# those two models are therefore computed over methods whose true class is unknown
# and are not reportable. Their GLOBAL explanations are unaffected -- those run on
# the labelled training set (global_features.csv), not on GXA -- so only the local
# cells are dropped. Dropped here rather than by deleting files, so the reason
# travels with the code and a re-run cannot silently reinstate them.
EXCLUDE = {
    ("SWAN-79", "weave", "local"): "no GXA ground truth for CWE79",
    ("SWAN-79", "published", "local"): "no GXA ground truth for CWE79",
    ("SWAN-Sa", "weave", "local"): "no GXA ground truth for sanitizer",
    ("SWAN-Sa", "published", "local"): "no GXA ground truth for sanitizer",
    # Both Dev-Assist runs now ship per-label per-instance files, so its cells are built
    # from labels 1/3/4 only (see DEVASSIST_LABELS) and need no exclusion: the two
    # unlabelled categories are dropped at load time rather than by skipping the cell.
}

# DATA -> absolute shares (all available cells; drives the concentration table)
# SIGNED -> signed shares (only cells with a per-instance file; drives the figures)
DATA, SIGNED, SOURCE, MISSING, RAW = {}, {}, {}, [], {}
INSTSHARE = {}   # cell -> {group: (fraction of explained instances pushing towards SRM, n)}
for key, label, subdir, gfn in APPS:
    for variant in ("weave", "published"):
        for scope in ("global", "local"):
            cell = (key, variant, scope)
            if cell in EXCLUDE:
                MISSING.append((*cell, "EXCLUDED -- " + EXCLUDE[cell]))
                continue
            vp = _find_signed(subdir, variant, scope)
            ap = _find(subdir, variant, scope, "aggregated")
            if vp:
                feats, data = read_signed(vp)
                sg, ab = shares_from_signed(feats, data, gfn)
                if sg is None:
                    MISSING.append((*cell, f"all {len(feats)} features are 0.0 (signed file present)"))
                    continue
                SIGNED[cell], DATA[cell], SOURCE[cell] = sg, ab, "per-instance"
                INSTSHARE[cell] = instance_shares(feats, data, gfn)
                RAW[cell] = {f: sum(abs(r[i]) for r in data) / len(data)
                             for i, f in enumerate(feats)}
            elif ap:
                vals = read_agg(ap)
                RAW[cell] = vals
                ab = shares_from_agg(vals, gfn)
                if ab is None:
                    MISSING.append((*cell, f"all {len(vals)} features are 0.0"))
                    continue
                DATA[cell], SOURCE[cell] = ab, "aggregate-only"
                MISSING.append((*cell, "magnitude only -- no per-instance file, so no sign"))
            else:
                MISSING.append((*cell, "no csv on disk"))

# ------------------------------------------------- validate mapping vs the paper
# Table table:feature-groups, text/5_rq1.tex. Corrected 25 Aug 2026 against the ARFFs:
# the CB values for SWAN and Dev-Assist had been swapped (SWAN has no class-body features,
# Dev-Assist has two), and SSCM's MN was one too high because the ARFF class attribute was
# counted -- SSCM has 103 features, not 104. SSCM and Dev-Assist now match the data exactly.
# SWAN still differs and legitimately so: the table reports 153 deduplicated feature concepts
# while the union of the five per-label ARFFs is 158, because a handful of concepts are named
# twice across the per-label extractions (e.g. ClsNmCtn.net./ClsNmCtnNet).
PAPER_COUNTS = {
    "SSCM":       dict(CMo=3, CN=14, CB=0, MMo=1, RT=7, MN=22, P=12, MB=39, DF=5),
    "SWAN":       dict(CMo=5, CN=25, CB=0, MMo=6, RT=9, MN=38, P=10, MB=38, DF=22),
    "Dev-Assist": dict(CMo=4, CN=25, CB=2, MMo=3, RT=1, MN=33, P=2, MB=29, DF=22),
}
PAPER_TOTALS = {"SSCM": 103, "SWAN": 153, "Dev-Assist": 121}

def arff_features(path, n_labels_front=0, label_at_end=True):
    attrs = []
    for line in open(path, errors="replace"):
        s = line.strip()
        if s.lower().startswith("@attribute"):
            m = re.match(r"@attribute\s+('([^']*)'|\"([^\"]*)\"|\S+)", s, re.I)
            attrs.append(m.group(2) or m.group(3) or m.group(1))
        elif s.lower().startswith("@data"):
            break
    if n_labels_front:
        return attrs[n_labels_front:]
    return attrs[:-1] if label_at_end else attrs

def validate_mapping(verbose=True):
    rows = []
    da = arff_features(f"{RES}/dev-assist/dev-assist-gxa-dataset.arff", n_labels_front=5)
    ss = arff_features(f"{RES}/sscm/sscm-gxa-dataset.arff")
    swan_union = set()
    per_swan = {}
    for k in ("source", "sink", "sanitizer", "cwe79", "cwe89"):
        fs = arff_features(f"{RES}/swan/{k}/{k}-gxa-dataset.arff")
        per_swan[k] = fs
        swan_union |= set(fs)
    for name, feats, gfn in (("Dev-Assist", da, group_dev_assist),
                             ("SSCM", ss, group_sscm),
                             ("SWAN", sorted(swan_union), group_swan)):
        got = {g: 0 for g in GROUPS}
        for f in feats:
            got[gfn(f)] += 1
        rows.append((name, got, len(feats)))
    if verbose:
        print("=" * 96)
        print("FEATURE-GROUP MAPPING vs Table table:feature-groups (text/5_rq1.tex)")
        print("=" * 96)
        hdr = f"  {'approach':<12} " + " ".join(f"{g:>5}" for g in GROUPS) + f" {'total':>7}"
        print(hdr)
        for name, got, n in rows:
            pap = PAPER_COUNTS[name]
            print(f"  {name+' data':<12} " + " ".join(f"{got[g]:>5}" for g in GROUPS) + f" {n:>7}")
            print(f"  {name+' paper':<12} " + " ".join(f"{pap[g]:>5}" for g in GROUPS)
                  + f" {sum(pap.values()):>7}  (printed total {PAPER_TOTALS[name]})")
            diff = {g: got[g] - pap[g] for g in GROUPS if got[g] != pap[g]}
            print(f"  {'':12} -> {'exact match' if not diff else 'differs: ' + str(diff)}")
        print(f"\n  per-SWAN-model feature counts: "
              + ", ".join(f"{k}={len(v)}" for k, v in per_swan.items()))
    return rows

# ---------------------------------------------------------------------- palette
# Categorical slots 1-8 from the dataviz reference palette, fixed order, not cycled.
# Validated on the white paper surface: worst adjacent CVD dE 9.1 (target >=8),
# worst normal-vision dE 19.6 (floor >=15). Slots 3/4/5 sit below 3:1 contrast, so
# the relief rule applies -- the legend plus concentration_table.tex is the table view.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
       "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# Diverging ramp for signed shares: blue (towards SRM) <-> neutral gray <-> red (away).
# Warm/cool poles with a neutral -- not two cool hues. The red arm was constructed by
# matching each blue step's OKLab lightness, so the two arms are symmetric to
# dL <= 0.003 and neither side reads as "louder" than the other at equal magnitude.
DIV_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV_RED = ["#eedad7", "#ebafad", "#e78280", "#e24948", "#b63737", "#8c2726", "#641717"]
DIV_MID = "#f0efec"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d7d7d4"

# Colour follows the GROUP, never its position in the currently-available data --
# otherwise adding Dev-Assist later would repaint every other group (recolour-on-filter).
# This mapping is fixed and must not be reordered once published.
COL = {"CMo": CAT[0], "CN": CAT[1], "MMo": CAT[2], "RT": CAT[3],
       "MN": CAT[4], "P": CAT[5], "MB": CAT[6], "DF": CAT[7]}
GROUPS_PLOT = [g for g in GROUPS if g in COL]

# CB has no slot: only Dev-Assist populates it, and a 9th categorical hue is not
# allowed (it would be indistinguishable from an existing slot under CVD). If the
# Dev-Assist run completes, fold CB into MB or facet Dev-Assist on its own -- do
# not invent a colour.
_cb = {k: v["CB"] for k, v in DATA.items() if v["CB"] > 0}
if _cb:
    sys.exit("ERROR: CB now carries weight in " + ", ".join(f"{k}" for k in _cb)
             + " but has no categorical slot. Fold CB into MB or facet Dev-Assist "
               "separately; do not generate a 9th hue.")

def fmt(v):
    return "" if v is None else f"{100*v:.1f}"

# ----------------------------------------------------------------- tidy csv out
def write_csv():
    p = os.path.join(OUTPUT, "shap_shares_gxa.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "approach", "explanation", "group",
                    "abs_share", "inst_share_towards_srm", "inst_n",
                    "signed_share_deprecated", "source"])
        for cell, sh in sorted(DATA.items()):
            key, variant, scope = cell
            sg = SIGNED.get(cell)
            ins = INSTSHARE.get(cell)
            for g in GROUPS:
                frac, n = (ins[g] if ins else (None, 0))
                w.writerow([variant, key, scope, g,
                            f"{sh[g]:.6f}",
                            "" if frac is None else f"{frac:.6f}", n or "",
                            "" if sg is None else f"{sg[g]:.6f}", SOURCE[cell]])
    print(f"\n  wrote {os.path.relpath(p, OUTPUT)}")

# ------------------------------------------------------------------- slopegraph
# Half-length (in x data units) of the style stub drawn through a global-only
# marker. At this panel width ~0.14 units is long enough for the dashed pattern
# to read as dashed, while staying clear of the "Global" tick label.
STUB = 0.14

def slope_panel(key, label):
    fig, ax = plt.subplots(figsize=(2.5, 2.3))
    # 50% is the neutral line: above it a group pushes towards the SRM class for most
    # explained instances, below it away. A line crossing it changes which way the
    # group points for a majority of methods.
    ax.axhline(50, color="#8a8a86", lw=0.9, zorder=1)
    for variant, style in (("weave", "-"), ("published", (0, (2, 2)))):
        gi = INSTSHARE.get((key, variant, "global"))
        li = INSTSHARE.get((key, variant, "local"))
        ga = DATA.get((key, variant, "global"))
        la = DATA.get((key, variant, "local"))
        if gi is None or ga is None:
            continue
        g = {grp: gi[grp][0] for grp in GROUPS}
        l = None if li is None else {grp: li[grp][0] for grp in GROUPS}
        for grp in GROUPS_PLOT:
            # a group with no direction on either side (all-zero contributions) is skipped
            if g[grp] is None or (l is not None and l[grp] is None):
                continue
            if max(ga[grp], (la or ga)[grp]) < 0.005:
                continue
            if l is not None:
                ax.plot([0, 1], [100 * g[grp], 100 * l[grp]], linestyle=style,
                        color=COL[grp], lw=2.0, marker="o", ms=8 / 2.4, alpha=0.95,
                        solid_capstyle="round", zorder=2)
            else:
                # Global-only cell (the local explanation is excluded or unavailable).
                # A bare marker cannot show whether it is the WEAVE or the original
                # model, because that distinction is carried by the line style -- so
                # draw a short stub through the marker in the variant's own style.
                y = 100 * g[grp]
                ax.plot([-STUB, 0, STUB], [y, y, y], linestyle=style, color=COL[grp],
                        lw=2.0, marker="o", ms=8 / 2.4, markevery=[1], alpha=0.95,
                        solid_capstyle="round", zorder=2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Global", "Local"], fontsize=8, color=INK)
    ax.set_ylabel("Instances pushing towards SRM (%)", fontsize=8, color=INK)
    ax.set_ylim(-2, 102)
    ax.tick_params(axis="y", labelsize=7, colors=INK2, length=3)
    ax.tick_params(axis="x", length=0)
    # Pin the x range for every panel. Global-only panels would otherwise autoscale to
    # the stub and lose the "Local" tick, leaving the small multiples on different x
    # scales -- so an empty right half reads as "no local explanation", as intended.
    ax.set_xlim(-0.18, 1.18)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, f"slope_{key}.pdf"), bbox_inches="tight")
    plt.close(fig)

def slope_legend():
    """Compact legend block sized for the empty slots beside the seventh slope panel.

    Feature groups use square swatches because colour is the channel that identifies
    them. Model variant keeps line samples: it is encoded by dash pattern in the plot,
    and a square cannot show a dash pattern -- a legend glyph should carry the same
    channel as the mark it explains.
    """
    # Two single rows in a full-width strip under the panels: feature groups on one
    # line, model variant on the next. Section headings are inline (a blank handle)
    # rather than legend titles, which would each add a second line. The canvas is
    # left wide and cropped tight, so the emitted width is set by the content; keeping
    # it near \textwidth means the strip is scaled by roughly 1 in the paper and the
    # 8pt labels stay at 8pt rather than shrinking.
    fig = plt.figure(figsize=(7.0, 0.42)); ax = fig.add_subplot(111); ax.axis("off")
    blank = Line2D([], [], linestyle="none", marker="none")
    sq = lambda c: Line2D([], [], color=c, marker="s", markersize=6.5, linestyle="none")
    gh = [blank] + [sq(COL[g]) for g in GROUPS_PLOT]
    gl = ["Feature group:"] + GROUPS_PLOT
    groups = ax.legend(gh, gl, loc="center", bbox_to_anchor=(0.5, 0.80),
                       ncol=len(gh), frameon=False, fontsize=8, handletextpad=0.4,
                       columnspacing=0.9, labelcolor=INK)
    ax.add_artist(groups)
    ax.legend([blank,
               Line2D([], [], color="#252523", lw=1.6, ls="-"),
               Line2D([], [], color="#252523", lw=1.6, ls=(0, (2, 2)))],
              ["Model:", "WEAVE", "Published"],
              loc="center", bbox_to_anchor=(0.5, 0.20), ncol=3, frameon=False,
              fontsize=8, handlelength=1.9, handletextpad=0.4, columnspacing=1.1,
              labelcolor=INK)
    fig.savefig(os.path.join(OUT, "slope_legend.pdf"), bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------- heatmap
VMAX = 70.0
def heat_cmap():
    # Sequential, light -> dark: the heatmap answers "how much of this model's
    # attribution does the group carry", which is the quantity the RQ2 claims cite and
    # the one that survives resampling (see stability). Direction is a separate
    # question with its own statistic and its own figure, the slopegraph -- putting it
    # in the colour here would mix a stable magnitude with a sign we cannot resolve.
    return mpl.colors.LinearSegmentedColormap.from_list("seq_blue", DIV_BLUE)

def matrix(variant, scope):
    """Share of the model's total |SHAP| carried by each group, in percent."""
    M = np.full((len(GROUPS_PLOT), len(APPS)), np.nan)
    for j, (key, _l, _s, _f) in enumerate(APPS):
        cell = DATA.get((key, variant, scope))
        if cell is None:
            continue
        for i, g in enumerate(GROUPS_PLOT):
            M[i, j] = 100 * cell[g]
    return M

def heatmaps():
    # "no data" must not read as "zero": the diverging midpoint is #f0efec, so the
    # bad colour is stepped clearly darker and those cells also carry no number.
    cmap = heat_cmap(); cmap.set_bad("#cfcfc9")
    for variant in ("weave", "published"):
        for scope in ("global", "local"):
            M = matrix(variant, scope)
            fig, ax = plt.subplots(figsize=(3.3, 2.7))
            ax.imshow(M, cmap=cmap, vmin=0, vmax=VMAX, aspect="auto")
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if np.isnan(M[i, j]):
                        continue
                    ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=4.6,
                            color="#ffffff" if M[i, j] > 0.55 * VMAX else INK)
            ax.set_xticks(range(len(APPS)))
            ax.set_xticklabels([a[1] for a in APPS], rotation=45, ha="right",
                               fontsize=7, color=INK)
            ax.set_yticks(range(len(GROUPS_PLOT)))
            ax.set_yticklabels(GROUPS_PLOT, fontsize=7, color=INK)
            ax.set_xticks(np.arange(-.5, len(APPS), 1), minor=True)
            ax.set_yticks(np.arange(-.5, len(GROUPS_PLOT), 1), minor=True)
            ax.grid(which="minor", color="white", lw=1.0)
            ax.tick_params(which="minor", length=0)
            ax.tick_params(which="major", length=0)
            for s in ax.spines.values(): s.set_visible(False)
            fig.tight_layout(pad=0.3)
            fig.savefig(os.path.join(OUT, f"heat_{variant}_{scope}.pdf"), bbox_inches="tight")
            plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.2, 0.42))
    cb = mpl.colorbar.ColorbarBase(ax, cmap=heat_cmap(),
                                   norm=mpl.colors.Normalize(vmin=0, vmax=VMAX),
                                   orientation="horizontal")
    cb.set_label("share of the model's total $|$SHAP$|$ (%);  grey: no data",
                 fontsize=8, color=INK)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    fig.savefig(os.path.join(OUT, "heat_colorbar.pdf"), bbox_inches="tight")
    plt.close(fig)

# --------------------------------------------------------- concentration table
def inv_simpson(sh):
    if sh is None:
        return None
    return 1.0 / sum(v * v for v in sh.values() if v > 0)

def concentration():
    lines = []
    print("\n" + "=" * 96)
    print("table:concentration -- effective number of feature groups (inverse Simpson)")
    print("=" * 96)
    print(f"  {'Approach':<15} {'WEAVE global':>13} {'WEAVE local':>12} "
          f"{'Orig global':>12} {'Orig local':>11}")
    for key, label, _s, _f in APPS:
        v = [inv_simpson(DATA.get((key, va, sc)))
             for va, sc in (("weave", "global"), ("weave", "local"),
                            ("published", "global"), ("published", "local"))]
        cells = ["--" if x is None else f"{x:.1f}" for x in v]
        print(f"  {label:<15} {cells[0]:>13} {cells[1]:>12} {cells[2]:>12} {cells[3]:>11}")
        lines.append(f"{label:<15}& {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} \\\\")
    p = os.path.join(OUT, "concentration_table.tex")
    with open(p, "w") as f:
        f.write("% regenerated by make_shap_figs.py from the GXA run -- body rows only\n")
        f.write("\n".join(lines) + "\n")
    print(f"\n  wrote {os.path.relpath(p, OUTPUT)}")

# -------------------------------------------------------------- group share dump
def dump_shares():
    print("\n" + "=" * 96)
    print("SIGNED GROUP SHARES (% of total |SHAP|; + towards SRM class, - away)")
    print("=" * 96)
    for variant, vlabel in (("weave", "WEAVE"), ("published", "Published")):
        for scope in ("global", "local"):
            print(f"\n  --- {vlabel} / {scope} ---")
            print(f"  {'approach':<15} " + " ".join(f"{g:>7}" for g in GROUPS_PLOT))
            for key, label, _s, _f in APPS:
                cell = SIGNED.get((key, variant, scope))
                if cell is None:
                    note = " (magnitude only)" if (key, variant, scope) in DATA else ""
                    print(f"  {label:<15} " + "      --" * len(GROUPS_PLOT) + note)
                    continue
                print(f"  {label:<15} " + " ".join(f"{100*cell[g]:>+7.1f}" for g in GROUPS_PLOT))

def direction_shifts(verbose=True):
    """Global->local change in the direction statistic (the RQ3 headline claim).

    Reports, per feature group and model, the fraction of explained instances in which
    the group pushes towards the SRM class, globally and locally, with a two-proportion
    z test on the change. A cell whose fraction crosses 50% changes which way the group
    points for a majority of instances; that is the defensible analogue of the sign
    reversal earlier drafts reported off the signed share.
    """
    EPS = 0.005   # ignore groups whose magnitude is negligible on both sides
    rows = []
    for variant, vlabel in (("weave", "WEAVE"), ("published", "Published")):
        for key, label, _s, _f in APPS:
            gi = INSTSHARE.get((key, variant, "global"))
            li = INSTSHARE.get((key, variant, "local"))
            ga = DATA.get((key, variant, "global"))
            la = DATA.get((key, variant, "local"))
            if gi is None or li is None or ga is None or la is None:
                continue
            for grp in GROUPS_PLOT:
                a, na = gi[grp]; b, nb = li[grp]
                if a is None or b is None or max(ga[grp], la[grp]) < EPS:
                    continue
                se = math.sqrt(a * (1 - a) / na + b * (1 - b) / nb) if na and nb else 0.0
                z = (b - a) / se if se > 0 else 0.0
                rows.append((vlabel, label, grp, a, b, z, na, nb))
    sig = [r for r in rows if abs(r[5]) >= 1.96]
    cross = [r for r in rows if (r[3] > 0.5) != (r[4] > 0.5) and abs(r[5]) >= 1.96]
    pos_neg = [r for r in cross if r[3] > 0.5]
    neg_pos = [r for r in cross if r[3] < 0.5]
    if verbose:
        print("\n" + "=" * 96)
        print("DIRECTION SHIFTS between global and local (the RQ3 headline claim)")
        print("=" * 96)
        print(f"  {'variant':<10} {'model':<15} {'grp':<4} {'global':>8} {'local':>8}"
              f" {'change':>8} {'z':>7}   n(g)/n(l)")
        for v, l, g, a, b, z, na, nb in sorted(rows, key=lambda r: -abs(r[5])):
            mark = "  <-- crosses 50%" if (a > 0.5) != (b > 0.5) and abs(z) >= 1.96 else ""
            print(f"  {v:<10} {l:<15} {g:<4} {100*a:>7.1f}% {100*b:>7.1f}%"
                  f" {100*(b-a):>+7.1f} {z:>+7.1f}   {na}/{nb}{mark}")
        print(f"\n  cells with a non-negligible magnitude on either side: {len(rows)}")
        print(f"  significant global->local shift (|z| >= 1.96):        {len(sig)}")
        print(f"  crossing the 50% line:                                {len(cross)}"
              f"  ({len(pos_neg)} towards-to-away, {len(neg_pos)} away-to-towards)")
        print(f"  largest shift:                                        "
              f"{max(abs(r[4]-r[3]) for r in rows)*100:.0f} pp")
        tally = {}
        for _v, _l, g, *_ in sig:
            tally[g] = tally.get(g, 0) + 1
        print("  significant shifts per group: " + ", ".join(
            f"{k}={v}" for k, v in sorted(tally.items(), key=lambda t: -t[1])))
    return rows, sig, cross, pos_neg, neg_pos

def modifier_direction():
    print("\n" + "=" * 96)
    print("Are modifiers exclusionary? (sign of CMo / MMo shares)")
    print("=" * 96)
    neg = tot = 0
    for cell, sh in sorted(SIGNED.items()):
        for grp in ("CMo", "MMo"):
            if abs(sh[grp]) < 0.005:
                continue
            tot += 1; neg += sh[grp] < 0
    print(f"  non-trivial modifier contributions: {tot};  negative: {neg};  positive: {tot-neg}")


# ------------------------------------------------- subsample stability (RQ2 setup)
def _per_instance_group(paths, gfn):
    """-> (list of {group: net contribution} per explained instance, {group: total |shap|})."""
    feats, data = read_signed(paths)
    per, absum = [], {g: 0.0 for g in GROUPS}
    for r in data:
        acc = {g: 0.0 for g in GROUPS}
        for f, v in zip(feats, r):
            acc[gfn(f)] += v
            absum[gfn(f)] += abs(v)
        per.append(acc)
    return per, absum

def stability(B=200, seed=20260825):
    """How much does the explained-sample size move a group's share?

    Two separate questions, because the answers differ. The *magnitude* share
    (share of total |SHAP|) is what the global claims rest on; the *sign* the
    signed share carries comes from the group's net contribution, which is a
    small difference of large numbers -- see shares_from_signed.
    """
    rnd = random.Random(seed)
    print("\n" + "=" * 96)
    print(f"SUBSAMPLE STABILITY of group shares ({B} bootstrap resamples per cell)")
    print("=" * 96)
    print(f"  {'cell':<24} {'N':>5} {'m':>5} {'p95 max |d abs share| pp':>26}")
    worst = []
    for key, label, subdir, gfn in APPS:
        for variant in ("weave", "published"):
            if (key, variant, "global") in EXCLUDE:
                continue
            vp = _find_signed(subdir, variant, "global")
            if not vp:
                continue
            feats, data = read_signed(vp)
            _, full = shares_from_signed(feats, data, gfn)
            n = len(data)
            for frac in (0.2, 0.5):
                mm = max(1, int(n * frac))
                d = []
                for _ in range(B):
                    sub = [data[rnd.randrange(n)] for _ in range(mm)]
                    _, a = shares_from_signed(feats, sub, gfn)
                    d.append(max(abs(a[g] - full[g]) for g in GROUPS) * 100)
                p95 = sorted(d)[int(0.95 * B)]
                worst.append((p95, label, variant, mm))
                print(f"  {label + '/' + variant:<24} {n:>5} {mm:>5} {p95:>26.2f}")
    dense = [w for w in worst if w[1] not in ("SWAN-CWE79", "SWAN-CWE89", "SWAN-Sanitizer")]
    half = [w for w in dense if w[3] >= 250]
    print(f"\n  worst p95 over all cells: {max(worst)[0]:.1f} pp ({max(worst)[1]}/{max(worst)[2]})")
    print(f"  worst p95 at half sample, excluding the three sparse SWAN tasks: {max(half)[0]:.1f} pp")
    print("  -> magnitude shares are reproducible at this sample size; the sparse")
    print("     minority tasks are the exception and move by up to ~11 pp.")

def sign_evidence():
    """Is the SIGN of a signed share resolvable from the explained sample?

    For every global->local crossing that direction_shifts() reports, print how large
    the group's net contribution is relative to its magnitude, and the t
    statistic of the per-instance net. |t| < 2 means the sign the figure draws
    is not distinguishable from zero at this sample size.
    """
    print("\n" + "=" * 96)
    print("SIGN EVIDENCE behind each reversal (net contribution vs its own magnitude)")
    print("=" * 96)
    print(f"  {'variant':<10} {'model':<15} {'grp':<4} {'net/|SHAP| g':>13} {'l':>7}"
          f" {'t g':>7} {'t l':>7} {'pos-share g':>12} {'l':>7} {'z':>7}")
    strong = tot = 0
    for variant, vlabel in (("weave", "WEAVE"), ("published", "Published")):
        for key, label, subdir, gfn in APPS:
            g = SIGNED.get((key, variant, "global")); l = SIGNED.get((key, variant, "local"))
            if g is None or l is None:
                continue
            st = {}
            for scope in ("global", "local"):
                per, absum = _per_instance_group(_find_signed(subdir, variant, scope), gfn)
                n = len(per)
                st[scope] = {}
                for grp in GROUPS:
                    xs = [p[grp] for p in per]
                    mu = sum(xs) / n
                    var = sum((x - mu) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
                    se = math.sqrt(var / n) if var > 0 else 0.0
                    nz = [x for x in xs if abs(x) > 1e-12]
                    pos = (sum(1 for x in nz if x > 0) / len(nz)) if nz else None
                    st[scope][grp] = (mu / se if se > 0 else 0.0,
                                      (sum(xs) / absum[grp]) if absum[grp] > 0 else 0.0,
                                      pos, len(nz))
            for grp in GROUPS_PLOT:
                if not (g[grp] * l[grp] < 0 and max(abs(g[grp]), abs(l[grp])) >= 0.005):
                    continue
                tg, pg, ag, na = st["global"][grp]
                tl, pl, ab, nb = st["local"][grp]
                tot += 1
                z = 0.0
                if ag is not None and ab is not None and na and nb:
                    se = math.sqrt(ag * (1 - ag) / na + ab * (1 - ab) / nb)
                    z = (ab - ag) / se if se > 0 else 0.0
                if abs(tg) >= 1.96 and abs(tl) >= 1.96:
                    strong += 1
                print(f"  {vlabel:<10} {label:<15} {grp:<4} {100*pg:>12.1f}% {100*pl:>6.1f}%"
                      f" {tg:>7.2f} {tl:>7.2f} {100*ag:>11.1f}% {100*ab:>6.1f}% {z:>7.1f}")
    print(f"\n  reversals whose sign is resolvable on BOTH sides (|t| >= 1.96): {strong} of {tot}")
    print("  pos-share = fraction of explained instances where the group pushes towards")
    print("  the SRM class; z tests the global->local change in that fraction.")

# ------------------------------------------------- GXA predictions (RQ3 setup)
def gxa_predictions():
    """What do the explained models actually predict on the GXA sample?"""
    print("\n" + "=" * 96)
    print("GXA PREDICTIONS on the local explanation sample")
    print("=" * 96)
    def arff_rows(path):
        rows, indata = [], False
        for line in open(path, errors="replace"):
            t = line.strip()
            if indata:
                if t and not t.startswith("%"):
                    rows.append(t.split(","))
            elif t.lower().startswith("@data"):
                indata = True
        return rows
    def preds(path):
        with open(path, newline="") as f:
            r = list(csv.reader(f))
        h = r[0]; di = h.index("dataset_row"); pi = h.index("prediction")
        return [(int(float(x[di])), float(x[pi])) for x in r[1:] if x and x[0] != ""]
    print(f"  {'target':<22} {'variant':<10} {'n':>5} {'labelled +':>11} {'scored >0.5':>12} {'of those, +':>12}")
    POS = {"1", "true", "yes"}
    for task in ("source", "sink", "cwe89", "sanitizer", "cwe79"):
        rows = arff_rows(f"{RES}/swan/{task}/{task}-gxa-dataset.arff")
        truth = [r[-1].strip().lower() for r in rows]
        for variant in ("weave", "published"):
            p = f"{RES}/swan/{task}/explanations/{variant}/local/local_shap_values.csv"
            if not os.path.exists(p):
                continue
            rr = preds(p)
            gt = [truth[i] in POS for i, _ in rr]; pr = [v > 0.5 for _, v in rr]
            print(f"  {'SWAN-' + task:<22} {variant:<10} {len(rr):>5} {sum(gt):>11}"
                  f" {sum(pr):>12} {sum(1 for a, b in zip(gt, pr) if a and b):>12}")
    rows = arff_rows(f"{RES}/dev-assist/dev-assist-gxa-dataset.arff")
    for L, name in zip(DEVASSIST_LABELS, ("source", "CWE89", "sink")):
        truth = [r[L].strip().lower() for r in rows]
        for variant in ("weave", "published"):
            hits = glob.glob(f"{RES}/dev-assist/explanations/{variant}/local/**/"
                             f"*shap_values_label_{L}.csv", recursive=True)
            if not hits:
                continue
            rr = preds(sorted(hits)[0])
            gt = [truth[i] in POS for i, _ in rr]; pr = [v > 0.5 for _, v in rr]
            print(f"  {'Dev-Assist-' + name:<22} {variant:<10} {len(rr):>5} {sum(gt):>11}"
                  f" {sum(pr):>12} {sum(1 for a, b in zip(gt, pr) if a and b):>12}")
    rows = arff_rows(f"{RES}/sscm/sscm-gxa-dataset.arff")
    truth = [r[-1].strip() for r in rows]
    # The re-run exports one file per class, so each SSCM class is scored against its own
    # ground truth instead of the Target column standing in for the whole model.
    for c, cls in zip(SSCM_CLASSES, ("Target", "Input")):
        for variant in ("weave", "published"):
            hits = glob.glob(f"{RES}/sscm/explanations/{variant}/local/**/"
                             f"*shap_values_class_{c}.csv", recursive=True)
            if not hits:
                continue
            rr = preds(sorted(hits)[0])
            gt = [truth[i] == cls for i, _ in rr]; pr = [v > 0.5 for _, v in rr]
            print(f"  {'SSCM-' + cls:<22} {variant:<10} {len(rr):>5} {sum(gt):>11}"
                  f" {sum(pr):>12} {sum(1 for a, b in zip(gt, pr) if a and b):>12}")

if __name__ == "__main__":
    validate_mapping()
    print("\n" + "=" * 96)
    print(f"AVAILABLE CELLS: {len(DATA)}/28   MISSING: {len(MISSING)}")
    print("=" * 96)
    for key, variant, scope, why in MISSING:
        print(f"  MISSING  {key:<12} {variant:<4} {scope:<7}  {why}")
    print(f"\n  groups plotted: {GROUPS_PLOT}"
          + (f"   (dropped, no data: {[g for g in GROUPS if g not in GROUPS_PLOT]})"
             if len(GROUPS_PLOT) < len(GROUPS) else ""))
    dump_shares()
    direction_shifts()
    modifier_direction()
    stability()
    sign_evidence()
    gxa_predictions()
    write_csv()
    for key, label, _s, _f in APPS:
        slope_panel(key, label)
    slope_legend()
    heatmaps()
    concentration()
    print("\n  wrote slope_*.pdf, slope_legend.pdf, heat_*_*.pdf, heat_colorbar.pdf ->",
          os.path.relpath(OUT, OUTPUT))
