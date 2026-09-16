#!/usr/bin/env bash
# Step-6 runner (sep16 plan): #128 thread-2 — bare STOP entry lifecycle probe.
# No-fill (trigger >=5% above market), watch 420s. Usage:
#   run_bare_stop_probe.sh            -> DRY RUN (prints intended order, places nothing)
#   run_bare_stop_probe.sh --yes      -> place + watch (GTD, the 09-15 cancelled shape)
#   run_bare_stop_probe.sh --yes --duration DAY   -> the A/B variant (09-15 survivor shape)
set -u
cd "$(dirname "$0")/../../../.."
PY=.venv/bin/python
LOGDIR=plugins/dnse/testing/live_test/logs; mkdir -p "$LOGDIR"
TS=$(date +%H%M%S)

if printf '%s\n' "$@" | grep -q -- '--yes'; then
    echo "=== gates before a REAL placement: token + flat + L0 ==="
    $PY plugins/dnse/tools/token_status.py 2>&1 | tail -2
    $PY plugins/dnse/tools/venue.py flat >/dev/null 2>&1; FLAT=$?
    [ "$FLAT" -ne 0 ] && { echo "!!! NOT flat/clean ($FLAT) — ABORTING."; exit "$FLAT"; }
    $PY plugins/dnse/testing/live_test/level0_venue_semantics/l0_order_semantics.py \
        2>&1 | tail -3
    L0=${PIPESTATUS[0]}; [ "$L0" -ne 0 ] && { echo "!!! L0 FAILED ($L0) — ABORTING."; exit "$L0"; }
fi

$PY plugins/dnse/testing/live_test/probe_bare_stop_lifecycle.py "$@" \
    2>&1 | tee "$LOGDIR/bare_stop_$TS.log"
RC=${PIPESTATUS[0]}
echo "probe exit=$RC  (0 answered / 2 indeterminate — see VERDICT above)  log=$LOGDIR/bare_stop_$TS.log"
exit "$RC"
