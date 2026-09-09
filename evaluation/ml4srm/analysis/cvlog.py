#!/usr/bin/env python3
"""Read the repeated-cross-validation log out of the experimenter's SQL dump.

`inputs/model-selection-dump.sql` is a mysqldump of the jaicore-experimenter database that
the empirical evaluator writes to.

Four tables. Only two are needed:

    swan          1,600 rows covering six single-label tasks, keyed by the ARFF path
    dev_assist      300 rows, the multi-label MEKA task
    sscm, cwe79     verbatim subsets of `swan`, so they are ignored

Two shapes in the data are worth knowing, because both would corrupt an average if handled
naively:

*Each configuration is recorded twice.* Every (task, model family) group holds 200 rows over
100 distinct seeds, the same experiments written under two different `dataset` strings, one an
absolute path and one just "weave". Rows are therefore deduplicated on (task, family, seed),
which is verified: within every group a seed's metrics are identical across its copies.

*A task does not identify a model.* Each task carries two configurations, the published model
and the WEAVE one, so the caller names the model family it wants rather than selecting by task.
"""

import collections
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP = os.path.join(HERE, "inputs", "model-selection-dump.sql")

TASKS = ("cwe79", "cwe89", "sink", "source", "sanitizer", "sscm")

# Longest-first, so that a descriptor naming an inner classifier is attributed to its
# outer one: LWL wraps RandomForest, EnsembleML wraps PS which wraps J48 or LMT.
FAMILIES = (
    "ClassificationViaRegression", "RandomSubSpace", "SimpleLogistic", "LogitBoost",
    "EnsembleML-J48", "EnsembleML-LMT", "LWL", "SMO", "Bagging", "RandomForest", "LC",
)

_ESCAPES = {"n": "\n", "r": "\r", "t": "\t", "0": "\0",
            "\\": "\\", "'": "'", '"': '"', "Z": "\x1a", "b": "\b"}


def _values(sql):
    """Split one mysqldump VALUES list into rows of already-unescaped fields."""
    rows, cur, field = [], [], []
    depth, inq, esc = 0, False, False
    for c in sql:
        if inq:
            if esc:
                field.append(_ESCAPES.get(c, c))
                esc = False
            elif c == "\\":
                esc = True
            elif c == "'":
                inq = False
            else:
                field.append(c)
        elif c == "'":
            inq = True
        elif c == "(" and depth == 0:
            depth, cur, field = 1, [], []
        elif c == "," and depth == 1:
            cur.append("".join(field))
            field = []
        elif c == ")" and depth == 1:
            cur.append("".join(field))
            rows.append(cur)
            depth = 0
        elif depth == 1:
            field.append(c)
    return rows


def _columns(sql, table):
    m = re.search(r"CREATE TABLE `" + table + r"` \((.*?)\n\) ENGINE", sql, re.S)
    if not m:
        raise KeyError(f"no CREATE TABLE for `{table}` in the dump")
    return re.findall(r"^\s*`([A-Za-z0-9_]+)`\s+\w", m.group(1), re.M)


def read_table(table, path=DUMP):
    """-> [row dict] for one table of the dump, column names as in the schema."""
    sql = open(path, encoding="utf-8", errors="replace").read()
    names = _columns(sql, table)
    rows = []
    for stmt in re.findall(r"INSERT INTO `" + table + r"` VALUES (.*?);\n", sql, re.S):
        rows += _values(stmt)
    bad = [r for r in rows if len(r) != len(names)]
    if bad:
        raise ValueError(f"`{table}`: {len(bad)} rows do not match the {len(names)} columns")
    return [dict(zip(names, r)) for r in rows]


def task_of(row):
    """Which detection target a row belongs to, from the ARFF path it was run on."""
    for field in ("datasetWeave", "dataset"):
        m = re.search(r"/(" + "|".join(TASKS) + r")/", row.get(field) or "")
        if m:
            return m.group(1)
    return None


def family_of(row):
    """The model family, named so that a Table 3 row can ask for the one it wants."""
    c = row.get("classifier") or ""
    # The multi-label wrappers are named by their inner classifier, since that is what
    # distinguishes the two Dev-Assist pipelines; everything else by its own class.
    if "EnsembleML" in c:
        return "EnsembleML-J48" if "J48" in c else "EnsembleML-LMT"
    if "multilabel.LC" in c:
        return "LC"
    return next((f for f in FAMILIES if f in c), c[:32])


def experiments(path=DUMP, verbose=False):
    """-> {(task, family): [row dict]}, one row per seed.

    `swan` and `dev_assist` only, since the other two tables duplicate `swan`. Deduplication
    is on (task, family, seed) and is checked rather than assumed: a seed whose copies
    disagree is reported instead of being silently collapsed.
    """
    groups = collections.defaultdict(dict)
    conflicts = []
    for table, task_fn in (("swan", task_of), ("dev_assist", lambda _r: "dev-assist")):
        for row in read_table(table, path):
            task = task_fn(row)
            if task is None:
                continue
            key = (task, family_of(row))
            seed = row.get("seeds")
            prior = groups[key].get(seed)
            if prior is not None:
                if prior.get("avgPrecision") != row.get("avgPrecision"):
                    conflicts.append((key, seed))
                continue
            groups[key][seed] = row

    out = {k: list(v.values()) for k, v in groups.items()}
    if verbose:
        print("=" * 78)
        print(f"CROSS-VALIDATION LOG  {os.path.basename(path)}")
        print("=" * 78)
        print(f"  {'task':12} {'model family':30} {'seeds':>6}")
        for k in sorted(out):
            print(f"  {k[0]:12} {k[1]:30} {len(out[k]):>6}")
        odd = [k for k, v in out.items() if len(v) != 100]
        if odd:
            print(f"  note: {len(odd)} group(s) do not have 100 seeds: {odd}")
        if conflicts:
            print(f"  WARNING: {len(conflicts)} seeds recorded twice with different metrics")
        print(f"  {len(out)} configurations, {sum(len(v) for v in out.values())} experiments")
    return out


if __name__ == "__main__":
    experiments(verbose=True)
