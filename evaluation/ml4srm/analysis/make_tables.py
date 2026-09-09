#!/usr/bin/env python3
"""Regenerate the five tables the paper reports, as LaTeX, from the shipped data.

Output: output/tables/table<N>_*.tex, one file per table.

    Table 1  table:srm-sources        per-source provenance of the 10,180 methods
    Table 2  table:feature-groups     feature groups and per-approach feature counts
    Table 3  table:selected-models    repeated cross-validation, published vs WEAVE
    Table 4  table:concentration      effective number of feature groups
    Table 5  table:gxa-predictions    what the explained models predict on the GXA sample

Table 1 needs the SRM-Dataset repository, which owns the catalog. Point SRM_DATASET_HOME at
a checkout of https://github.com/secure-software-engineering/srm-dataset to include it;
without it that one table is skipped and the other four are still written.

Numbers only. The descriptions and examples in Table 2 and every caption are editorial text
and are reproduced here verbatim so the emitted file is complete, not because they are
derived.
"""

import collections
import glob
import importlib.util
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
OUTPUT = os.environ.get("ML4SRM_FIGURES", os.path.join(HERE, "output"))
TABLES = os.path.join(OUTPUT, "tables")
INPUTS = os.path.join(HERE, "inputs")
WEAVE_ROOT = os.path.dirname(os.path.dirname(PKG))          # .../weave
SRM_DATASET = os.environ.get(                               # default: a sibling checkout
    "SRM_DATASET_HOME", os.path.join(os.path.dirname(WEAVE_ROOT), "srm-dataset"))


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tex_int(n):
    """LaTeX thousands separator, as the paper writes them."""
    s = f"{n:,}".replace(",", "{,}")
    return s


def write(name, body):
    os.makedirs(TABLES, exist_ok=True)
    p = os.path.join(TABLES, name)
    with open(p, "w") as f:
        f.write(body if body.endswith("\n") else body + "\n")
    print(f"  wrote tables/{name}")


# ------------------------------------------------------------------ Table 1
# Roles the paper does not model. A method whose only role is one of these counts as not
# security-relevant, on both sides, because no model here is trained to predict it.
NOT_SRM_ROLES = {"propagator", "authentication", "authenticator", "auth",
                 "auth-safe-state", "auth-unsafe-state", "auth-no-change"}

# Source label -> (display name with its citation, block). The dataset's `discovery`
# strings are matched case-insensitively as substrings, since a few carry a suffix.
SOURCES = [
    ("find-sec-bugs",   r"find-sec-bugs",                          "prior"),
    ("sscm",            r"SSCM",                                   "prior"),
    ("swan",            r"SWAN",                                   "prior"),
    ("thecodemaster",   r"TheCodeMaster",                          "prior"),
    ("taintbench",      r"TaintBench~\cite{taintbench}",           "new"),
    ("codoc",           r"CoDoC~\cite{samhi2023negative}",         "new"),
    ("secucheck",       r"SecuCheck~\cite{secucheck}",             "new"),
    ("owasp",           r"OWASP Benchmark~\cite{owaspbench}",      "new"),
]


def table1():
    cat = os.path.join(SRM_DATASET, "datasets", "java", "srm-dataset-java.json")
    if not os.path.isfile(cat):
        print("  skipped table1: the SRM-Dataset catalog is not available")
        print(f"    looked in {cat}")
        print("    set SRM_DATASET_HOME to a checkout to include it")
        return False

    methods = json.load(open(cat))["methods"]
    counts = {k: collections.Counter() for k, _, _ in SOURCES}
    totals = collections.Counter()
    unmatched = 0
    for m in methods:
        disc = (m.get("discovery") or "").strip().lower()
        key = next((k for k, _, _ in SOURCES if k in disc), None)
        if key is None:
            unmatched += 1
            continue
        roles = {r.strip().lower() for r in (m.get("srm") or [])}
        modelled = roles - NOT_SRM_ROLES
        c = counts[key]
        for role, col in (("source", "So"), ("sink", "Si"), ("sanitizer", "Sa")):
            if role in modelled:
                c[col] += 1
                totals[col] += 1
        if not modelled:
            c["notsrm"] += 1
            totals["notsrm"] += 1
        if m.get("cwe"):
            c["CWE"] += 1
            totals["CWE"] += 1
        c["n"] += 1
        totals["n"] += 1

    if unmatched:
        print(f"  note: {unmatched} methods have a discovery value matching no known source")

    rows = []
    for block, heading in (("prior", r"\emph{Prior SRM list~\cite{ide-workshop-paper}}"),
                           ("new", r"\emph{Newly added sources}")):
        rows.append(r"\multicolumn{7}{l}{" + heading + r"} \\")
        for key, label, blk in SOURCES:
            if blk != block:
                continue
            c = counts[key]
            rows.append(
                rf"\quad {label} & {tex_int(c['So'])} & {tex_int(c['Si'])} & {tex_int(c['Sa'])}"
                rf" & {tex_int(c['notsrm'])} & {tex_int(c['CWE'])}"
                rf" & \textbf{{{tex_int(c['n'])}}} \\")
        rows.append(r"\midrule")

    # how many of the not-SRM methods are there only because of an unmodelled role
    only_unmodelled = sum(
        1 for m in methods
        if (roles := {r.strip().lower() for r in (m.get("srm") or [])})
        and not (roles - NOT_SRM_ROLES))

    body = rf"""% generated by analysis/make_tables.py -- do not edit by hand
\begin{{table}}[ht]
    \centering

 \begin{{tabular}}{{lrrrrrr}}
\toprule
\textbf{{Source}}  & \textbf{{So}} & \textbf{{Si}} & \textbf{{Sa}} & \textbf{{$\notin_{{SRM}}$}} & \textbf{{CWE}} & \textbf{{Total}} \\
\midrule
{chr(10).join(rows)}
\textbf{{Total}} & {tex_int(totals['So'])} & {tex_int(totals['Si'])} & {tex_int(totals['Sa'])} & {tex_int(totals['notsrm'])} & {tex_int(totals['CWE'])} & \textbf{{{tex_int(totals['n'])}}} \\
\bottomrule
\end{{tabular}}
\end{{table}}"""
    write("table1_srm_sources.tex", body)
    return True


# ------------------------------------------------------------------ Table 2
# Group id -> (bold lead-in, the rest of the description, example). Editorial text,
# reproduced so the emitted float is complete; only the counts are derived.
GROUP_ROWS = [
    ("C",   None,          r"\textit{\textbf{Class Declarations}}", "", ""),
    ("CMo", r"Modifier",   "Type of class modifier.", "Is \"public\"", "row"),
    ("CN",  r"Name",       "Bag-of-words approach checks if tokens in the vocabulary appear "
                           "as prefixes, suffixes, or substrings.", "Ends with \"servlet\"", "row"),
    ("CB",  r"Body",       "Quantifiable metrics for the class.", "Lines of code", "row"),
    ("M",   None,          r"\textit{\textbf{Method Definitions}}", "", ""),
    ("MMo", r"Modifier",   r"Method modifiers. \textit{See CMo}.", "Is private", "row"),
    ("RT",  r"Return Type", "Evaluates the data type of values returned by the method.",
                           "Is char[]", "row"),
    ("MN",  r"Name",       r"Method name. \textit{See CN}.", "Starts with \"do\"", "row"),
    ("P",   r"Parameter",  r"Evaluates parameter names (\textit{See CN}), data types, and "
                           r"counts.", "Is string", "row"),
    ("MB",  r"Body",       "Software metrics and method invocations in the method body.",
                           "Invokes sink SRM", "row"),
    ("DF",  r"Data-Flow",  "Data-flow among method components with static analysis.",
                           "Flows to sink method", "row"),
]


def table2(sf):
    mapped = sf.validate_mapping(verbose=False)
    counted = {name: got for name, got, _n in mapped}
    totals = {name: n for name, _got, n in mapped}

    # The counts are derived, so say where they part company with what the paper prints.
    # SWAN is the known case: its five per-label models carry different feature sets, and
    # the union across them is wider than the single set the paper's column reports.
    for name, got, n in mapped:
        pap = sf.PAPER_COUNTS[name]
        diff = {g: got[g] - pap[g] for g in sf.GROUPS if got[g] != pap[g]}
        if diff:
            print(f"  WARNING table2: {name} counts differ from the manuscript by {diff}")
            print(f"    derived total {n}, manuscript prints {sf.PAPER_TOTALS[name]}")
            if name == "SWAN":
                print("    expected: the derived figure is the union over the five per-label")
                print("    ARFFs (source 68, sink 84, sanitizer 48, cwe79 26, cwe89 22), which")
                print("    is wider than one model's set. Decide which the column should report.")

    lines = []
    for gid, lead, desc, example, kind in GROUP_ROWS:
        if kind != "row":
            lines.append(rf"\textit{{\textbf{{{gid}}}}}  & {desc} &  &  &  \\")
            continue
        c = [counted["SSCM"][gid], counted["SWAN"][gid], counted["Dev-Assist"][gid]]
        lines.append(
            rf"\textit{{\textbf{{{gid}}}}}  & \textit{{\textbf{{{lead}}}}} {desc}"
            rf" & {example} & {c[0]} & {c[1]} & {c[2]} \\")

    t = [totals["SSCM"], totals["SWAN"], totals["Dev-Assist"]]
    body = rf"""% generated by analysis/make_tables.py -- do not edit by hand
% Counts are derived from the ARFF headers; descriptions and examples are editorial.
\begin{{table}}[ht]
\begin{{tabular}}{{lp{{5.5cm}}lccc}}
\toprule
&   & & \multicolumn{{3}}{{c}}{{\textbf{{Feature Count}}}}\\ \cmidrule{{4-6}}
\textbf{{ID}}   & \textbf{{Feature Groups}} & \textbf{{Example}} & \multicolumn{{1}}{{l}}{{\textbf{{\cite{{sas}}}}}} & \multicolumn{{1}}{{l}}{{\textbf{{\cite{{swan}}}}}} & \multicolumn{{1}}{{l}}{{\textbf{{\cite{{ide-workshop-paper}}}}}}   \\
\midrule

{chr(10).join(lines)}
\cmidrule{{4-6}}
  & & & {t[0]} & {t[1]} & {t[2]}\\
     \bottomrule
\end{{tabular}}
\end{{table}}"""
    write("table2_feature_groups.tex", body)


# ------------------------------------------------------------------ Table 3
# (row label, task, WEAVE family, published family, WEAVE abbreviation, published abbreviation)
# The dump keeps the superseded AutoML candidates beside the selected ones -- LWL for cwe89,
# ClassificationViaRegression for sink, LC for Dev-Assist -- so each row names the family it
# wants rather than taking "whichever is not SMO".
TABLE3_ROWS = [
    ("Dev-Assist",     "dev-assist", "EnsembleML-J48", "EnsembleML-LMT", "PS+J48", "PS+LMT"),
    ("SSCM",           "sscm",       "Bagging",        "SMO",            "B+RF",   "SVM"),
    ("SWAN-CWE79",     "cwe79",      "RandomSubSpace", "SMO",            "RSS+L",  "SVM"),
    ("SWAN-CWE89",     "cwe89",      "SimpleLogistic", "SMO",            "SL",     "SVM"),
    ("SWAN-Sanitizer", "sanitizer",  "Bagging",        "SMO",            "B+L",    "SVM"),
    ("SWAN-Sink",      "sink",       "LogitBoost",     "SMO",            "LB+RF",  "SVM"),
    ("SWAN-Source",    "source",     "RandomForest",   "SMO",            "RF",     "SVM"),
]


def _f1(p, r):
    """Table 3's macro F1: the F1 of the macro-averaged precision and recall.

    This is *not* the mean of the per-class F1 scores, which is the more usual reading of
    "macro F1" and which differs materially -- by 0.11 for the published SWAN-CWE79 model.
    The definition here is the one that reproduces the published table. If the paper's
    wording is ever tightened, this is the line to revisit.
    """
    return 2 * p * r / (p + r) if (p + r) else 0.0


def _wilcoxon_p(a, b):
    """Two-sided Wilcoxon signed-rank p, normal approximation with a continuity correction.

    Sample size here is 100 matched seeds, well inside the range where the approximation is
    appropriate, so this avoids a scipy dependency for one test.
    """
    d = [x - y for x, y in zip(a, b) if x != y]
    n = len(d)
    if n < 10:
        return 1.0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w = sum(r for r, v in zip(ranks, d) if v > 0)
    mu = n * (n + 1) / 4
    sigma = (n * (n + 1) * (2 * n + 1) / 24) ** 0.5
    if sigma == 0:
        return 1.0
    z = (abs(w - mu) - 0.5) / sigma
    # two-sided normal tail
    return 2 * (1 - 0.5 * (1 + _erf(z / (2 ** 0.5))))


def _ranksum_p(a, b):
    """Two-sided Mann-Whitney / Wilcoxon rank-sum p, normal approximation with tie correction.

    Dev-Assist's published and WEAVE runs use different seeds, so the samples are independent
    and the signed-rank test above does not apply. This is the test the caption names.
    """
    n1, n2 = len(a), len(b)
    if n1 < 10 or n2 < 10:
        return 1.0
    pooled = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks = [0.0] * len(pooled)
    i = 0
    ties = 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        t = j - i + 1
        ties += t ** 3 - t
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    r1 = sum(r for r, (_v, g) in zip(ranks, pooled) if g == 0)
    u = r1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    n = n1 + n2
    var = n1 * n2 / 12 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0:
        return 1.0
    z = (abs(u - mu) - 0.5) / var ** 0.5
    return 2 * (1 - 0.5 * (1 + _erf(z / (2 ** 0.5))))


def _erf(x):
    # Abramowitz and Stegun 7.1.26, |error| < 1.5e-7
    sign = -1 if x < 0 else 1
    x = abs(x)
    t = 1 / (1 + 0.3275911 * x)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
              - 0.284496736) * t + 0.254829592) * t * pow(2.718281828459045, -x * x)
    return sign * y


def table3(mp, cvlog):
    groups = cvlog.experiments()

    def metrics(task, family):
        rows = groups.get((task, family))
        if not rows:
            sys.exit(f"  table3: no experiments for {task}/{family} in the dump")
        P = [mp.num(r.get("avgPrecision")) for r in rows]
        R = [mp.num(r.get("avgRecall")) for r in rows]
        P = [v for v in P if v is not None]
        R = [v for v in R if v is not None]
        # micro-F1: MEKA fills avgMicroF1Measure, WEKA leaves it NULL and fills microFMeasure
        mi = [mp.num(r.get("avgMicroF1Measure")) if mp.num(r.get("avgMicroF1Measure")) is not None
              else mp.num(r.get("microFMeasure")) for r in rows]
        mi = [v for v in mi if v is not None]
        # macro-F1: MEKA reports it directly; for the WEKA tasks it is derived, see _f1
        lbl = [mp.num(r.get("avgMacroF1MeasureLbl")) for r in rows]
        lbl = [v for v in lbl if v is not None]
        if lbl:
            fma, per_seed = statistics.fmean(lbl), lbl
        else:
            fma = _f1(statistics.fmean(P), statistics.fmean(R))
            per_seed = [_f1(p, r) for p, r in zip(P, R)]
        return {"P": statistics.fmean(P), "R": statistics.fmean(R), "Fma": fma,
                "Fmi": statistics.fmean(mi),
                # keyed by seed, because the two variants of a task are not stored in the
                # same seed order -- pairing them by position would compare unlike runs
                "by_seed": dict(zip([r.get("seeds") for r in rows], per_seed))}

    lines = []
    for label, task, fam_w, fam_p, abbr_w, abbr_p in TABLE3_ROWS:
        w, p = metrics(task, fam_w), metrics(task, fam_p)
        # Matched seeds get the signed-rank test, aligned on the seed rather than on
        # position. Dev-Assist's two runs share no seeds, so that row gets the unpaired
        # rank-sum test, as the caption states.
        shared = sorted(set(w["by_seed"]) & set(p["by_seed"]))
        if len(shared) >= 0.9 * min(len(w["by_seed"]), len(p["by_seed"])):
            pv = _wilcoxon_p([w["by_seed"][k] for k in shared],
                             [p["by_seed"][k] for k in shared])
        else:
            pv = _ranksum_p(list(w["by_seed"].values()), list(p["by_seed"].values()))
        mark = ""
        if pv < 0.01:
            mark = r"$^{\uparrow}$" if w["Fma"] > p["Fma"] else r"$^{\downarrow}$"

        pub_model = abbr_p
        if label == "SSCM":
            pub_model = r"\multicolumn{1}{c}{\multirow{6}{*}{SVM}}"
        elif label.startswith("SWAN"):
            pub_model = ""
        lines.append(
            f"{label} & {abbr_w} & {w['P']:.2f} & {w['R']:.2f} & {w['Fma']:.2f}{mark}"
            f" & {w['Fmi']:.2f} & {pub_model} & {p['P']:.2f} & {p['R']:.2f}"
            f" & {p['Fma']:.2f} & {p['Fmi']:.2f} " + r"\\")

    body = r"""% generated by analysis/make_tables.py -- do not edit by hand
\begin{table*}[ht]
\centering
  \begin{threeparttable}[ht]
 \small\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccccccccc}
\toprule
& \multicolumn{5}{c}{\textit{WEAVE Models}} &  \multicolumn{5}{c}{\textit{Published Models}} \\ \cmidrule(lr){2-6} \cmidrule(lr){7-11}

\textbf{Approach}  & \textbf{Model} & \textbf{P} & \textbf{R} & \textbf{F1}\textit{\textsubscript{ma}} & \textbf{F1}\textit{\textsubscript{mi}}   & \textbf{Model} & \textbf{P} & \textbf{R} & \textbf{F1}\textit{\textsubscript{ma}} & \textbf{F1}\textit{\textsubscript{mi}} \\
\midrule

""" + "\n".join(lines) + r""" \hline

\end{tabular}
  \end{threeparttable}
\end{table*}"""
    write("table3_selected_models.tex", body)


# ------------------------------------------------------------------ Table 4
def table4(sf):
    lines = []
    for key, label, _s, _f in sf.APPS:
        cells = []
        for variant in ("weave", "published"):
            for scope in ("global", "local"):
                d = sf.DATA.get((key, variant, scope))
                cells.append(f"{sf.inv_simpson(d):.1f}" if d else "--")
        lines.append(f"{label:<14} & " + " & ".join(cells) + r" \\")

    body = r"""% generated by analysis/make_tables.py -- do not edit by hand
\begin{table}[ht]
\centering
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{\textbf{\weave}} & \multicolumn{2}{c}{\textbf{Published}} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
\textbf{Approach} & \textbf{Global} & \textbf{Local} & \textbf{Global} & \textbf{Local} \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}"""
    write("table4_concentration.tex", body)


# ------------------------------------------------------------------ Table 5
def table5(sf):
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
        import csv
        with open(path, newline="") as f:
            r = list(csv.reader(f))
        di, pi = r[0].index("dataset_row"), r[0].index("prediction")
        return [(int(float(x[di])), float(x[pi])) for x in r[1:] if x and x[0] != ""]

    POS = {"1", "true", "yes"}
    got = {}

    def add(name, variant, truth, rr):
        gt = [truth[i] for i, _ in rr]
        pr = [v > 0.5 for _, v in rr]
        got[(name, variant)] = (sum(gt), sum(pr),
                                sum(1 for a, b in zip(gt, pr) if a and b))

    for task, label in (("source", "SWAN-Source"), ("sink", "SWAN-Sink"),
                        ("cwe89", "SWAN-CWE89")):
        truth = [r[-1].strip().lower() in POS
                 for r in arff_rows(f"{sf.RES}/swan/{task}/{task}-gxa-dataset.arff")]
        for variant in ("weave", "published"):
            p = f"{sf.RES}/swan/{task}/explanations/{variant}/local/local_shap_values.csv"
            if os.path.exists(p):
                add(label, variant, truth, preds(p))

    rows = arff_rows(f"{sf.RES}/dev-assist/dev-assist-gxa-dataset.arff")
    for L, name in zip(sf.DEVASSIST_LABELS, ("Source", "CWE89", "Sink")):
        truth = [r[L].strip().lower() in POS for r in rows]
        for variant in ("weave", "published"):
            hits = glob.glob(f"{sf.RES}/dev-assist/explanations/{variant}/local/**/"
                             f"*shap_values_label_{L}.csv", recursive=True)
            if hits:
                add(f"Dev-Assist-{name}", variant, truth, preds(sorted(hits)[0]))

    rows = arff_rows(f"{sf.RES}/sscm/sscm-gxa-dataset.arff")
    for c, cls in zip(sf.SSCM_CLASSES, ("Target", "Input")):
        truth = [r[-1].strip() == cls for r in rows]
        for variant in ("weave", "published"):
            hits = glob.glob(f"{sf.RES}/sscm/explanations/{variant}/local/**/"
                             f"*shap_values_class_{c}.csv", recursive=True)
            if hits:
                add(f"SSCM-{cls}", variant, truth, preds(sorted(hits)[0]))

    ORDER = [["SWAN-Source", "SWAN-Sink", "SWAN-CWE89"],
             ["Dev-Assist-Source", "Dev-Assist-Sink", "Dev-Assist-CWE89"],
             ["SSCM-Input", "SSCM-Target"]]
    lines = []
    for block in ORDER:
        for name in block:
            w = got.get((name, "weave"))
            p = got.get((name, "published"))
            if not w or not p:
                continue
            lines.append(f"{name:<18}& {w[0]:>3} & {w[1]:>3} & {w[2]:>2}"
                         f" & {p[1]:>3} & {p[2]:>2} " + r"\\")
        lines.append(r"\midrule")
    lines.pop()

    body = r"""% generated by analysis/make_tables.py -- do not edit by hand
\begin{table}[ht]
\centering
\begin{tabular}{lrrrrr}
\toprule
& & \multicolumn{2}{c}{\textbf{\weave}} & \multicolumn{2}{c}{\textbf{Published}} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
\textbf{Model} & \textbf{Pos.} & \textbf{Fired} & \textbf{TP} & \textbf{Fired} & \textbf{TP} \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}"""
    write("table5_gxa_predictions.tex", body)


def main():
    print("=" * 78)
    print(f"TABLES -> {os.path.relpath(TABLES, HERE)}")
    print("=" * 78)
    sf = load("make_shap_figs")
    mp = load("make_perf_tables")
    table1()
    table2(sf)
    table3(mp, load("cvlog"))
    table4(sf)
    table5(sf)
    print("\n  Each file is a complete float. Diff against the manuscript to see what moved.")


if __name__ == "__main__":
    sys.exit(main())
