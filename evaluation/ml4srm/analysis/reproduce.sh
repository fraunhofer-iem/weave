#!/bin/sh
# Regenerate the paper's figures and tables from the shipped explanation outputs.
#
# This is the recompute path: it turns the committed SHAP outputs into the results the
# paper reports. It does not re-run WEKA or the SHAP explainer. For
# that see "Reproducing a run" in ../README.md, which needs a cluster.
#
# Outputs land in ./output: figures in output/rq1 and output/rq2, and the five tables the
# paper reports in output/tables. Set ML4SRM_FIGURES to collect them elsewhere, and
# ML4SRM_DATA to read the explanations from a different copy.
#
# Table 3 comes from the experimenter SQL dump in inputs/; Table 1 from the SRM-Dataset catalog, which lives in its own repository. Set
# SRM_DATASET_HOME to a checkout of
# https://github.com/secure-software-engineering/srm-dataset to include it; without it that
# one table is skipped and everything else is still written.
set -e

cd "$(dirname "$0")"
OUT="${ML4SRM_FIGURES:-$PWD/output}"
LOG="$OUT/logs"
mkdir -p "$LOG"

PY="${PYTHON:-python3}"
$PY - <<'EOF' || { echo "Missing Python dependencies. Install with: pip install -r ../requirements.txt" >&2; exit 1; }
import importlib.util, sys
sys.exit(1 if [m for m in ("matplotlib", "numpy") if not importlib.util.find_spec(m)] else 0)
EOF

for s in make_perf_tables make_shap_figs make_tables; do
  printf '  %-20s' "$s"
  if $PY "$s.py" > "$LOG/$s.log" 2>&1; then
    echo "ok   (log: logs/$s.log)"
  else
    echo "FAILED"
    tail -20 "$LOG/$s.log" >&2
    exit 1
  fi
done

echo
echo "Tables written to $OUT/tables. Diff them against the manuscript to see what moved:"
ls "$OUT/tables" 2>/dev/null | sed 's/^/  /'
grep -h "^  WARNING" "$LOG/make_tables.log" 2>/dev/null || true
