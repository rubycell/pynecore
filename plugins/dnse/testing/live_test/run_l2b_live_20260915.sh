#!/usr/bin/env bash
# One-shot LIVE runner for l2b_fill_protect_flatten (2026-09-15, 1m, run-as-is).
# FILL TIER — takes a REAL position. Operator watches; DNSE app open for one-tap flatten.
# Sequence: pre-flight (token+flat) -> L0 gate -> l2b --broker @1m -> grade -> flat check.
# Exercises today's #124 fix (bounded re-arm) on a real conditional OCO bracket — watch the
# venue record for the SL/TP RESTING after the fill.
set -u
cd "$(dirname "$0")/../../../.."            # repo root
PY=.venv/bin/python
PYNE=.venv/bin/pyne
LOGDIR=plugins/dnse/testing/live_test/logs
mkdir -p "$LOGDIR"
TS=$(date +%H%M%S)
LOG="$LOGDIR/l2b_live_$TS.log"

echo "=== [1/4] pre-flight: token GOOD + account FLAT ==="
$PY plugins/dnse/tools/token_status.py 2>&1 | tail -3
$PY plugins/dnse/tools/venue.py flat > /dev/null 2>&1; FLAT=$?
if [ "$FLAT" -ne 0 ]; then
    echo "!!! account is NOT provably flat/clean (venue.py flat exit $FLAT) — ABORTING."
    $PY plugins/dnse/tools/venue.py status 2>&1 | sed 's/\x1b\[[0-9;]*m//g'
    exit "$FLAT"
fi
echo "  account FLAT + clean."

echo "=== [2/4] L0 venue-semantics gate (must exit 0) ==="
$PY plugins/dnse/testing/live_test/level0_venue_semantics/l0_order_semantics.py \
    2>&1 | tee "$LOGDIR/l2b_l0_$TS.log"
L0=${PIPESTATUS[0]}
if [ "$L0" -ne 0 ]; then
    echo "!!! L0 FAILED (exit $L0) — ABORTING. Nothing placed."
    exit "$L0"
fi

echo "=== [3/4] LIVE l2b --broker @1m (timeout 540s; a breakout stop may not fill — no-fill = no exposure) ==="
echo "    watch for: STOP fill -> native OCO bracket CREATED -> (bracket/max-2c/double-check) flatten"
timeout 540 $PYNE run plugins/dnse/testing/live_test/l2b_fill_protect_flatten.py \
    dnse_broker:VN30F1M@1 --broker 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
echo "  pyne exit=$RC (124/143 = timeout-kill, expected if it idled after a clean round-trip)"

echo "=== [4/4] teardown: account MUST be flat + clean ==="
$PY plugins/dnse/tools/venue.py status 2>&1 | sed 's/\x1b\[[0-9;]*m//g'
$PY plugins/dnse/tools/venue.py flat; FLATEND=$?
echo
echo "================ l2b LIVE SUMMARY ================"
echo "L0 gate    : exit $L0"
echo "pyne run   : exit $RC"
echo "flat@end   : exit $FLATEND  (0 = truly flat+clean)"
echo "log        : $LOG"
if [ "$FLATEND" -ne 0 ]; then
    echo "!!! NOT FLAT AT END — a position or order may be resting."
    echo "!!! FLATTEN IN THE DNSE APP (or: venue.py cancel <id>). Do NOT leave it open."
fi
exit 0
