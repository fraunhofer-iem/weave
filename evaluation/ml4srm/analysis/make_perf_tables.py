#!/usr/bin/env python3
"""Regenerate the minority-class performance table (RQ1) from the raw experiment log.

Input : inputs/model-selection-dump.sql via cvlog.py -- one row per
        (approach x classifier x seed), 1400 rows. Its `info` column holds WEKA's raw
        "Detailed Accuracy By Class" block, from which per-class Precision, Recall,
        F-Measure, MCC, ROC-AUC and PRC-AUC are parsed.
Output: figures/minority_class_metrics.csv     tidy per-class metrics, all approaches
                                              (this is the replication artefact cited
                                              in Section subsec-eval-rq1-results)
        figures/rq1/minority_class_table.tex   LaTeX body rows -- NOT used in the paper
                                              any more, kept in case the numbers are
                                              ever tabulated again

WHY THIS EXISTS. The minority-class figures quoted in the RQ1 prose -- positive-class
precision, recall and F1, and PRC-AUC -- are not in any of the paper's tables; Table 3
reports only macro averages and micro-F1. This script derives them from the raw log so
the numbers in the text are reproducible rather than hand-transcribed, and it
regression-checks itself against the recall values the paper previously tabulated.

NOTE ON SYMBOLS. `avgPrecision` / `avgRecall` in the dump are MACRO averages over
classes -- verified here as mean(per-class value) for all 14 configurations. They are
what Table 3 reports as P and R. The quantities below are POSITIVE-CLASS only and are
written P+, R+, F1+ to keep the two apart.

Run: python3 make_perf_tables.py
"""
import csv, os, re, statistics, sys, zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))            # .../evaluation/ml4srm/analysis
PKG  = os.path.dirname(HERE)                                # .../evaluation/ml4srm

# One root for the shipped explanation outputs, one for this script's inputs, one for
# what it writes. Only DATA_ROOT is worth overriding, and only to point at a second copy
# of the outputs; ML4SRM_FIGURES exists so the paper build can collect the figures
# in place instead of copying them out of the package afterwards.
DATA_ROOT = os.environ.get("ML4SRM_DATA", PKG)
INPUTS    = os.path.join(HERE, "inputs")
OUTPUT    = os.environ.get("ML4SRM_FIGURES", os.path.join(HERE, "output"))

DUMP = os.path.join(INPUTS, "model-selection-dump.sql")    # the experimenter database
OUT_TEX = os.path.join(OUTPUT, "rq1", "minority_class_table.tex")
OUT_CSV = os.path.join(OUTPUT, "minority_class_metrics.csv")

# ------------------------------------------------------------------- parsing
def num(s):
    try:
        return float(str(s).replace(",", "."))
    except (ValueError, AttributeError):
        return None

# Column order of WEKA's "Detailed Accuracy By Class" block, as it appears in `info`.
METRICS = ["TPRate", "FPRate", "P", "R", "F1", "MCC", "ROC", "PRC"]


def per_class(info):
    """-> {class_label: {metric: value}} from 'Detailed Accuracy By Class'."""
    out = {}
    for line in (info or "").splitlines():
        s = line.strip()
        if not s or s.startswith(("TP Rate", "Weighted", "===")):
            continue
        t = s.split()
        if len(t) == 9:
            vals = [num(x) for x in t[:8]]
            if all(v is not None for v in vals):
                out[t[-1]] = dict(zip(METRICS, vals))
    return out

# --------------------------------------------------------------- pipeline map
# (row label, task, WEAVE model family, published model family)
# Each row names the model family it wants rather than taking "whichever is not SMO".
# The five binary SWAN tasks only. SSCM is excluded because it is three-class, so it has no
# single positive class to report, and Dev-Assist because it is multi-label. Table 3's seven
# rows are a different, longer list and live in make_tables.py.
PIPELINES = [
    ("SWAN-CWE79",     "cwe79",     "RandomSubSpace", "SMO"),
    ("SWAN-CWE89",     "cwe89",     "SimpleLogistic", "SMO"),
    ("SWAN-Sanitizer", "sanitizer", "Bagging",        "SMO"),
    ("SWAN-Sink",      "sink",      "LogitBoost",     "SMO"),
    ("SWAN-Source",    "source",    "RandomForest",   "SMO"),
]
POSITIVE = "1"          # SWAN tasks are binary with the SRM class labelled 1

def collect(groups):
    """-> {(task, family): {(class, metric): [values over seeds]}}

    `groups` is what cvlog.experiments() returns, so the keys are already
    (task, model family) and each list holds one value per seed.
    """
    D = defaultdict(lambda: defaultdict(list))
    for key, rows in groups.items():
        for r in rows:
            for cls, m in per_class(r.get("info")).items():
                for k, v in m.items():
                    D[key][(cls, k)].append(v)
            for f in ("avgPrecision", "avgRecall", "avgMacroF1MeasureLbl"):
                v = num(r.get(f))
                if v is not None:
                    D[key][("__", f)].append(v)
            # Micro-F1 sits in different columns for the two experimenter front ends:
            # MEKA fills avgMicroF1Measure, WEKA leaves it NULL and fills microFMeasure.
            # For a single-label multi-class run micro-P, micro-R and micro-F1 coincide.
            mi = num(r.get("avgMicroF1Measure"))
            if mi is None:
                mi = num(r.get("microFMeasure"))
            if mi is not None:
                D[key][("__", "microF1")].append(mi)
    return D

def mean(d, cls, metric):
    v = d.get((cls, metric))
    return statistics.fmean(v) if v else None

# ------------------------------------------------------------------- reporting
def check_macro(D):
    """Confirm avgPrecision/avgRecall really are macro averages (Table 3's P and R)."""
    print("=" * 92)
    print("Are avgPrecision / avgRecall macro averages over classes?")
    print("=" * 92)
    bad = 0
    for (a, v), d in sorted(D.items()):
        classes = sorted({c for c, _ in d if c != "__"})
        if not classes:
            continue
        for field, metric in (("avgPrecision", "P"), ("avgRecall", "R")):
            if ("__", field) not in d:
                continue
            macro = statistics.fmean([mean(d, c, metric) for c in classes])
            got = statistics.fmean(d[("__", field)])
            ok = abs(macro - got) < 0.005
            bad += not ok
            if not ok:
                print(f"  MISMATCH {a}/{v} {field}: macro={macro:.4f} vs {got:.4f}")
    print(f"  checked {len(D)} configurations -> {'all consistent' if not bad else f'{bad} MISMATCHES'}")

PUBLISHED_RECALL = {          # what the current table:posrecall prints, for regression-checking
    "SWAN-CWE79": (0.10, 0.08), "SWAN-CWE89": (0.56, 0.61),
    "SWAN-Sanitizer": (0.82, 0.77), "SWAN-Sink": (0.81, 0.67),
    "SWAN-Source": (0.96, 0.97),
}

def build(D):
    rows = []
    print("\n" + "=" * 92)
    print("MINORITY-CLASS METRICS (positive class, mean of 100 seeds)")
    print("=" * 92)
    hdr = (f"  {'approach':15s} {'variant':10s} {'model':6s} {'P+':>6s} {'R+':>6s} "
           f"{'F1+':>6s} {'PRC':>6s} {'F1mi':>6s}")
    print(hdr)
    for label, task, fam_weave, fam_pub in PIPELINES:
        rec = {"approach": label}
        for variant, fam in (("weave", fam_weave), ("published", fam_pub)):
            d = D.get((task, fam))
            if d is None:
                sys.exit(f"no experiments for {task}/{fam} -- check the dump")
            model = fam
            vals = {m: mean(d, POSITIVE, m) for m in ("P", "R", "F1", "PRC")}
            vals["F1mi"] = statistics.fmean(d[("__", "microF1")])
            vals["n"] = len(d[(POSITIVE, "R")])
            vals["model"] = model
            rec[variant] = vals
            print(f"  {label:15s} {variant:10s} {model:6s} {vals['P']:6.3f} {vals['R']:6.3f} "
                  f"{vals['F1']:6.3f} {vals['PRC']:6.3f} {vals['F1mi']:6.3f}")
        rows.append(rec)
    print("\n  regression check against the values currently printed in table:posrecall:")
    for r in rows:
        exp = PUBLISHED_RECALL[r["approach"]]
        got = (round(r["weave"]["R"], 2), round(r["published"]["R"], 2))
        print(f"    {r['approach']:15s} paper R+={exp}  recomputed={got}  "
              f"{'OK' if got == exp else 'CHANGED'}")
    return rows

def write_tex(rows):
    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    with open(OUT_TEX, "w") as f:
        f.write("% regenerated by make_perf_tables.py -- body rows only\n")
        # No model column: Table 3 already names the pipelines, and 11 columns keeps
        # this table the same width as that one.
        for r in rows:
            a, p = r["weave"], r["published"]
            f.write(f"{r['approach']:15s}& {a['P']:.2f} & {a['R']:.2f} & {a['F1']:.2f} & "
                    f"{a['PRC']:.2f} & {a['F1mi']:.2f} & "
                    f"{p['P']:.2f} & {p['R']:.2f} & {p['F1']:.2f} & "
                    f"{p['PRC']:.2f} & {p['F1mi']:.2f} \\\\\n")
    print(f"\n  wrote {os.path.relpath(OUT_TEX, OUTPUT)}")

def write_csv(D):
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["approach_key", "variant", "class", "precision", "recall",
                    "f1", "mcc", "roc_auc", "prc_auc", "n_seeds"])
        for (a, v), d in sorted(D.items()):
            for cls in sorted({c for c, _ in d if c != "__"}):
                w.writerow([a, v, cls] +
                           [f"{mean(d, cls, m):.6f}" for m in ("P", "R", "F1", "MCC", "ROC", "PRC")] +
                           [len(d[(cls, "R")])])
    print(f"  wrote {os.path.relpath(OUT_CSV, OUTPUT)}")

def sscm_note(D):
    """SSCM is 3-class; its minority class is None (i.e. NOT security-relevant)."""
    print("\n" + "=" * 92)
    print("SSCM (3-class) per-class metrics -- for the prose, not the table")
    print("=" * 92)
    for variant in ("weave", "published"):
        d = D.get(("sscm", variant))
        if not d:
            continue
        print(f"  {variant}:")
        for cls in sorted({c for c, _ in d if c != "__"}):
            print(f"     {cls:8s} P={mean(d,cls,'P'):.3f} R={mean(d,cls,'R'):.3f} "
                  f"F1={mean(d,cls,'F1'):.3f} PRC={mean(d,cls,'PRC'):.3f}")

if __name__ == "__main__":
    import cvlog
    groups = cvlog.experiments(DUMP, verbose=True)
    D = collect(groups)
    check_macro(D)
    table = build(D)
    sscm_note(D)
    write_tex(table)
    write_csv(D)
