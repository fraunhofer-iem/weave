#!/bin/bash
# Build the Python environments the SHAP jobs need.
#
# RUN THIS ON A LOGIN NODE, once, before submitting jobs. Compute nodes do not
# reliably have outbound HTTPS - a job that tried to install there failed with
#   SSLError(FileNotFoundError(2, 'No such file or directory'))
# and left a half-uninstalled pip behind. The job scripts reuse an environment
# whenever its .weave-deps-ok sentinel exists, so once these are built the jobs
# never need the network.
#
#   bash evaluation/ml4srm/setup-venvs.sh          # build whatever is missing
#   FORCE=1 bash evaluation/ml4srm/setup-venvs.sh  # rebuild everything

set -u

export WEAVE_HOME="${WEAVE_HOME:-$PC2PFS/hpc-prf-crnrw/asej-2026/weave}"
REQS="${WEAVE_HOME}/evaluation/ml4srm/requirements.txt"
FORCE="${FORCE:-0}"

case "${WEAVE_HOME}" in
  *'$'*) echo "ERROR: WEAVE_HOME contains an unexpanded variable: ${WEAVE_HOME}" >&2
         echo '       Pass an absolute path: export WEAVE_HOME="$PC2PFS/.../weave"' >&2
         exit 1 ;;
esac
[ -f "$REQS" ] || { echo "ERROR: requirements not found: $REQS" >&2; exit 1; }

echo "WEAVE_HOME = ${WEAVE_HOME}"
echo "requirements = ${REQS}"
echo

# Derived from the job scripts so the two cannot drift apart.
# \+ (one or more) rather than * so this script's own grep pattern, which ends at
# the bracket, does not match itself when the glob picks up setup-venvs.sh.
VENVS=$(grep -ho 'weave_env_[A-Za-z0-9_]\+' "${WEAVE_HOME}"/evaluation/ml4srm/*.sh | sort -u)
[ -n "$VENVS" ] || { echo "ERROR: no venv names found in the job scripts" >&2; exit 1; }

failed=0
for name in $VENVS; do
  VENV="${WEAVE_HOME}/${name}"
  if [ "$FORCE" != "1" ] && [ -f "$VENV/.weave-deps-ok" ]; then
    echo "ok      $name (already provisioned)"
    continue
  fi
  echo "building $name ..."
  rm -rf "$VENV"
  python -m venv "$VENV" || { echo "FAILED  $name (venv creation)" >&2; failed=1; continue; }
  # Deliberately no 'pip install --upgrade pip': it needs the network before the
  # real install and, when it fails midway, it removes the pip it was replacing.
  if "$VENV/bin/pip" install --quiet -r "$REQS"; then
    if "$VENV/bin/python" -c "import shap, numpy, pandas, matplotlib, requests" 2>/dev/null; then
      touch "$VENV/.weave-deps-ok"
      echo "ok      $name"
    else
      echo "FAILED  $name (installed but not importable)" >&2; failed=1
    fi
  else
    echo "FAILED  $name (pip install)" >&2; failed=1
  fi
done

echo
if [ "$failed" = 0 ]; then
  echo "All environments ready. Jobs will reuse them and will not need the network."
else
  echo "Some environments failed to build - see above." >&2
fi
exit "$failed"
