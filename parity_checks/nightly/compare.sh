#!/usr/bin/env bash
# Nightly parity comparison for one suite (GH #702). Called by
# .github/workflows/nightly-parity.yml once a suite's runner has written its report.
#
#   compare.sh SUITE KIND REPORT [extra `verdicts.py diff` args, e.g. --expect-backend cc]
#
#   1. extract REPORT into $NIGHTLY_OUT/SUITE/verdicts.json. That file is the candidate
#      baseline: committing it as baselines/SUITE.json accepts tonight's state.
#   2. diff it against the committed baseline, parity_checks/nightly/baselines/SUITE.json.
#   3. diff it against the previous scheduled run's verdicts for SUITE, when one can be
#      downloaded. This catches a fix that regresses before anyone re-baselined it.
#
# Both Markdown reports go to $GITHUB_STEP_SUMMARY. Exit 1 on any alert or on a
# missing report; a missing baseline is an alert unless ALLOW_MISSING_BASELINE is set
# (only while a baseline is being bootstrapped on a pull request).
set -uo pipefail

suite=$1
kind=$2
report=$3
shift 3

here=$(cd "$(dirname "$0")" && pwd)
out=${NIGHTLY_OUT:-nightly-out}/$suite
summary=${GITHUB_STEP_SUMMARY:-/dev/null}
mkdir -p "$out"

if [ ! -s "$report" ]; then
  echo "::error::$suite: no report at $report; the run did not finish."
  printf '### %s\n\n**No report**: the run did not finish. See the job log.\n\n' "$suite" >>"$summary"
  printf '{"suite": "%s", "label": "", "no_report": true, "n_alerts": 1}\n' "$suite" >"$out/diff-baseline.json"
  exit 1
fi
cp "$report" "$out/report.json"

prov=$(printf '{"run_id": "%s", "sha": "%s", "event": "%s"}' \
  "${GITHUB_RUN_ID:-local}" "${GITHUB_SHA:-}" "${GITHUB_EVENT_NAME:-local}")
if ! python3 "$here/verdicts.py" extract --kind "$kind" --suite "$suite" \
  --report "$report" --out "$out/verdicts.json" --provenance "$prov"; then
  echo "::error::$suite: could not extract verdicts from $report"
  exit 1
fi

rc=0
allow=()
if [ -n "${ALLOW_MISSING_BASELINE:-}" ]; then
  allow=(--allow-missing-baseline)
fi
python3 "$here/verdicts.py" diff \
  --baseline "$here/baselines/$suite.json" --fresh "$out/verdicts.json" \
  --label "vs committed baseline" \
  --json-out "$out/diff-baseline.json" --md-out "$out/diff-baseline.md" \
  ${allow[@]+"${allow[@]}"} "$@" >/dev/null || rc=1
if [ -f "$out/diff-baseline.md" ]; then
  cat "$out/diff-baseline.md" >>"$summary"
else
  echo "::error::$suite: the baseline comparison failed to run"
  rc=1
fi

# The previous scheduled run's verdicts for this suite, if any is still downloadable.
if [ -n "${GH_TOKEN:-}" ] && [ -n "${GITHUB_REPOSITORY:-}" ]; then
  ids=$(gh run list -R "$GITHUB_REPOSITORY" --workflow nightly-parity.yml --branch main \
    --event schedule --limit 7 --json databaseId,status \
    --jq '.[] | select(.status == "completed") | .databaseId' 2>/dev/null || true)
  for id in $ids; do
    [ "$id" = "${GITHUB_RUN_ID:-}" ] && continue
    prev=$(mktemp -d)
    if gh run download "$id" -R "$GITHUB_REPOSITORY" -n "nightly-$suite" -D "$prev" >/dev/null 2>&1 &&
      [ -f "$prev/verdicts.json" ]; then
      python3 "$here/verdicts.py" diff \
        --baseline "$prev/verdicts.json" --fresh "$out/verdicts.json" \
        --label "vs previous night (run $id)" \
        --json-out "$out/diff-prev.json" --md-out "$out/diff-prev.md" "$@" >/dev/null || rc=1
      [ -f "$out/diff-prev.md" ] && cat "$out/diff-prev.md" >>"$summary"
      break
    fi
  done
fi

exit $rc
