#!/usr/bin/env bash
# PHASE 1 (NO-FILL) runner for l2b_entry_update — verify the chasing conditional ENTRY's
# stopPrice UPDATES on DNSE each 1m candle, with a 5% no-fill pad (default padPct=5.0).
# Up to ~10 min. NO money at risk (entry sits 5% above market, never fills). Operator watches.
# Sequence: pre-flight (token+flat) -> L0 gate -> l2b_entry_update --broker @1m (600s) -> status.
set -u
cd "$(dirname "$0")/../../../.."
PY=.venv/bin/python
PYNE=.venv/bin/pyne
LOGDIR=plugins/dnse/testing/live_test/logs
mkdir -p "$LOGDIR"
TS=$(date +%H%M%S)
LOG="$LOGDIR/l2b_entry_update_$TS.log"

echo "=== [1/4] pre-flight: token GOOD + account FLAT ==="
$PY plugins/dnse/tools/token_status.py 2>&1 | tail -2
$PY plugins/dnse/tools/venue.py flat > /dev/null 2>&1; FLAT=$?
if [ "$FLAT" -ne 0 ]; then
    echo "!!! account NOT provably flat/clean (exit $FLAT) — ABORTING."
    $PY plugins/dnse/tools/venue.py status 2>&1 | sed 's/\x1b\[[0-9;]*m//g'
    exit "$FLAT"
fi
echo "  FLAT + clean."

echo "=== [2/4] L0 venue-semantics gate (must exit 0) ==="
$PY plugins/dnse/testing/live_test/level0_venue_semantics/l0_order_semantics.py \
    2>&1 | tee "$LOGDIR/l2b_eu_l0_$TS.log"
L0=${PIPESTATUS[0]}
[ "$L0" -ne 0 ] && { echo "!!! L0 FAILED (exit $L0) — ABORTING."; exit "$L0"; }

echo "=== [3/4] l2b_entry_update --broker @1m, padPct=5 NO-FILL, 600s ==="
echo "    watch: each candle -> 'dispatching ENTRY ... stop=<new>' + a cancel->wire of the old"
echo "    (that IS the #85 cancel+replace); a 'quarantined' line = the #124 venue-cancel, NOT an amend bug"
timeout 600 $PYNE run plugins/dnse/testing/live_test/l2b_entry_update.py \
    dnse_broker:VN30F1M@1 --broker 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
echo "  pyne exit=$RC (124/143 = 10-min timeout, expected)"

echo "=== [4/4] status (a resting no-fill conditional E is EXPECTED; it is ours, cleaned up next) ==="
$PY plugins/dnse/tools/venue.py status 2>&1 | sed 's/\x1b\[[0-9;]*m//g'
echo
echo "================ ENTRY-UPDATE SUMMARY ================"
echo "L0 gate : exit $L0"
echo "pyne    : exit $RC"
echo "log     : $LOG"
echo "the sequence of stop= values (did it move each candle?):"
sed 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -aoE "dispatching ENTRY[^[]*stop=[0-9.]+" | tail -15
echo "quarantine check:"
sed 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -aic "quarantined" | xargs echo "  quarantined lines:"
exit 0
